"""Capture chat renderings from a served checkpoint's own template."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

LOOKUP = {'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{'ticket':{'type':'integer'}},'required':['ticket']}}}
SEARCH = {'type':'function','function':{'name':'search_files','description':'Find files whose text matches "pattern" — <regex> & globs, naïve café ✓','parameters':{'type':'object','required':['pattern'],'properties':{
    'pattern':{'type':'string','description':'A regex.\nSecond line.'},
    'path':{'type':'string','default':'.'},
    'limit':{'type':'integer','minimum':1,'default':20},
    'threshold':{'type':'number','default':0.5,'maximum':1.0,'minimum':1e-05},
    'budget':{'type':'number','default':1e16},
    'kinds':{'type':'array','items':{'type':'string','enum':['file','dir']}},
    'options':{'type':'object','properties':{'case':{'type':'boolean','default':False},'after':{'type':['string','null'],'default':None}},'additionalProperties':False},
}}}}
ASK = {'role':'user','content':'Call lookup for ticket 413.'}
CALL = {'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'lookup','arguments':{'ticket':413}}}]}
RESULT = {'role':'tool','name':'lookup','content':'LARCH-481'}
SEARCHES = [
    {'role':'system','content':'You are terse.'},
    {'role':'user','content':'  Where is main defined? 🙂\n'},
    {'role':'assistant','content':'Searching.','reasoning_content':'Two calls cover it.','tool_calls':[
        {'id':'call_a','type':'function','function':{'name':'search_files','arguments':{'pattern':'fn\\s+main\n# "quoted" <tag> & é','path':'src','limit':5,'threshold':0.25,'kinds':['file'],'options':{'case':True,'after':None},'zeta':None,'alpha':1.0}}},
        {'id':'call_b','type':'function','function':{'name':'lookup','arguments':{}}},
    ]},
    {'role':'tool','tool_call_id':'call_a','content':'src/main.rs:1'},
    {'role':'tool','tool_call_id':'call_b','content':'{"ok": true}'},
]
THOUGHT = [{'role':'user','content':'What is 2+2?'}, {'role':'assistant','content':'Four.','reasoning_content':'2+2 is 4.'}, {'role':'user','content':'And 3+3?'}]
PARTS = [{'role':'user','content':[{'type':'text','text':'Look.'},{'type':'image'},{'type':'text','text':'What is it?'}]}]


def cases(ink):
    high = {'reasoning_effort':'high'} if ink else {}
    found = {
        'plain': dict(messages=[ASK]),
        'system': dict(messages=SEARCHES[:2]),
        'closed': dict(messages=SEARCHES[:2] + [{'role':'assistant','content':'Done.'}], continuation='closed'),
        'tools': dict(messages=[ASK], tools=[LOOKUP], variables=high),
        'tool_result': dict(messages=[ASK, CALL, RESULT, {'role':'user','content':'Reply with just the lookup result.'}], tools=[LOOKUP], variables=high),
        'tool_steps': dict(messages=SEARCHES, tools=[SEARCH, LOOKUP]),
        'tool_steps_then_user': dict(messages=SEARCHES + [{'role':'user','content':'Thanks.'}], tools=[SEARCH, LOOKUP]),
        'bare_tool': dict(messages=[ASK], tools=[SEARCH['function']]),
        'thought': dict(messages=THOUGHT),
        'parts': dict(messages=PARTS),
        'null_content': dict(messages=[ASK, dict(CALL, content=None, name=None), RESULT], tools=[LOOKUP]),
        'reasoning': dict(messages=[ASK, dict(CALL, reasoning='Ticket 413.'), RESULT], tools=[LOOKUP]),
    }
    if ink:
        for effort in ('none', 'medium', 0.5, None):
            found[f'effort_{effort}'] = dict(messages=SEARCHES[:2], variables={'reasoning_effort':effort})
        found['named_assistant'] = dict(messages=[ASK, dict(CALL, name='lookup'), RESULT], tools=[LOOKUP])
    else:
        found['preserved_thought'] = dict(messages=THOUGHT, variables={'preserve_thinking':True})
        found['inline_thought'] = dict(messages=[THOUGHT[0], {'role':'assistant','content':'<think>\n2+2 is 4.\n</think>\n\nFour.'}, CALL, RESULT], tools=[LOOKUP])
        found['no_thinking'] = dict(messages=[ASK], variables={'enable_thinking':False})
        found['numbered_parts'] = dict(messages=PARTS, variables={'add_vision_id':True})
        found['continued'] = dict(messages=THOUGHT[:1] + [{'role':'assistant','content':'The answer is'}], continuation='continue_last')
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=('qwen', 'inkling'))
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    path = args.checkpoint
    ink = args.family == 'inkling'
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    found = cases(ink)
    for case in found.values():
        continuation = case.setdefault('continuation', 'start_assistant')
        options = dict(case.setdefault('variables', {}), add_generation_prompt=continuation == 'start_assistant')
        if continuation == 'continue_last':
            options['continue_final_message'] = True
        case['rendered'] = tok.apply_chat_template(json.loads(json.dumps(case['messages'])), tools=case.get('tools'), tokenize=False, **options)
        case['token_ids'] = tok.encode(case['rendered'], add_special_tokens=False)
    files = [path / name for name in ('chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json')]
    print(json.dumps(dict(
        checkpoint=path.name,
        sha256={file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files if file.exists()},
        versions={p: metadata.version(p) for p in ('transformers', 'tokenizers', 'jinja2')},
        template=tok.chat_template,
        cases=found,
    ), ensure_ascii=False))


if __name__ == '__main__':
    main()
