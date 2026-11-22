"""Capture how the Python servers split a served checkpoint's output into reasoning, text and tool calls."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

LOOKUP = {'type':'function','function':{'name':'lookup','parameters':{'type':'object','properties':{'ticket':{'type':'integer'}},'required':['ticket']}}}
ANY = {f'any{i}':{'type':'any'} for i in range(12)}
CONVERT = {'type':'function','function':{'name':'convert','parameters':{'type':'object','properties':dict(ANY, **{
    'text':{'type':'string'}, 'kind':{'type':'enum'}, 'count':{'type':'integer'}, 'big':{'type':' INT64 '}, 'ratio':{'type':'number'}, 'share':{'type':'float'},
    'flag':{'type':'boolean'}, 'items':{'type':'array'}, 'config':{'type':'object'}, 'rows':{'type':'list[int]'},
    'maybe':{'type':['string','null']}, 'either':{'anyOf':[{'type':'string'},{'type':'object'}]}, 'empty':{},
})}}}
THINKING = '<|im_start|>user\nCall lookup for ticket 413.<|im_end|>\n<|im_start|>assistant\n<think>\n'
ANSWERING = THINKING + '\n</think>\n\n'
M, T, X, J, E = '<|message_model|>', '<|content_thinking|>', '<|content_text|>', '<|content_invoke_tool_json|>', '<|end_message|>'
ASKED = '<|message_user|><|content_text|>Call lookup for ticket 413.<|end_message|>'


def call(name, **parameters):
    return f'<tool_call>\n<function={name}>\n' + ''.join(f'<parameter={key}>\n{value}\n</parameter>\n' for key, value in parameters.items()) + '</function>\n</tool_call>'


def qwen_cases():
    lookup = call('lookup', ticket=413)
    def case(text, finish='stop', prompt=THINKING, tools=(LOOKUP, CONVERT)):
        return dict(prompt=prompt, tools=list(tools), text=text, finish=finish)
    return {
        'answer': case('Add them.\n</think>\n\nFour. 🙂'),
        'call': case('Need the ticket.\n</think>\n\n' + lookup),
        'calls': case('Two calls.\n</think>\n\nSearching.\n\n' + call('convert', text='fn\\s+main\n# "quoted" <tag> & é', count=5) + '\n' + lookup + '\nDone.'),
        'typed': case('</think>\n\n' + call('convert', text='  two\nlines  ', kind='NULL', count='7.0', big='-12', ratio='2.50', share='3.0', flag='True', items='[1, "a\nb", {"k": null}]',
            config='{"b": 1.0, "a": "é", "b": [true, 1e-05, 1e16]}', rows='[[1, 2], []]', maybe='42', either='{"x": [0.5]}', empty='7', other='5', text2='null')),
        'loose': case('</think>\n\n' + call('convert', count=' 1e3 ', ratio='1e16', share='.5', flag='yes', items='not json', config='[1, 2', either='5', kind='7', text='')),
        'any': case('</think>\n\n' + call('convert', any0='42', any1='main.rs', any2='true', any3='True', any4='None', any5='[1, "a", [2.5]]', any6='"quoted"', any7='{"k": 1}', any8='[true]', any9='-3', any10='False', any11='1e5')),
        'fraction_as_integer': case('</think>\n\n' + call('convert', count='7.5')),
        'word_as_integer': case('</think>\n\n' + call('convert', count='seven') + lookup),
        'json_call': case('</think>\n\n<tool_call>\n{"name": "lookup", "arguments": {"ticket": 413}}\n</tool_call>'),
        'nameless': case('</think>\n\n<tool_call>\n<function=>\n</function>\n</tool_call>'),
        'empty_call': case('</think>\n\n<tool_call></tool_call>after'),
        'blank_line_after_function': case('</think>\n\n' + lookup.replace('</function>\n', '</function>\n\n')),
        'tight': case('</think>\n\n<tool_call><function= lookup ><parameter=ticket>413</parameter><parameter= spaced >v</parameter></function></tool_call>'),
        'parameters': case('</think>\n\n<tool_call>\n<function=lookup>\n<parameter=ticket>\n1\n</parameter>\n<parameter=ticket>\n\n2\n\n</parameter>\n<parameter=>\nx\n</parameter>\n<parameter=note>a < b <parameter=c>\n</parameter>\n</function>\n</tool_call>'),
        'blank_lines': case('</think>\n\n<tool_call>\n<function=lookup>\n<parameter=note>\n\nx\n\n</parameter>\n</function>\n</tool_call>'),
        'unknown_tool': case('</think>\n\n' + call('missing', ticket=413)),
        'bare_tool': case('</think>\n\n' + lookup, tools=[LOOKUP['function']]),
        'no_tools': case('</think>\n\n' + lookup, tools=[]),
        'truncated_call': case('Need it.\n</think>\n\n<tool_call>\n<function=lookup>\n<parameter=ticket>\n41', 'length'),
        'unclosed_call': case('</think>\n\n' + lookup.removesuffix('</tool_call>')),
        'call_while_thinking': case('Call it.\n' + lookup + '\n\nDone.'),
        'still_thinking': case('Still thinking'),
        'thinking_cut': case('Still thinking <', 'length'),
        'partial_marker': case('ok\n</think>\n\na <'),
        'partial_marker_cut': case('ok\n</think>\n\na <', 'length'),
        'held_twice': case('</think>\n\na <<'),
        'held_markers': case('</think>\n\na << b <thin << <too'),
        'thinks_again': case('</think>\n\nA<think>B</think>C'),
        'not_thinking': case('Four.', prompt=ANSWERING),
        'not_thinking_call': case(lookup, prompt=ANSWERING),
        'stray_markers': case('A</think>B</tool_call>C', prompt=ANSWERING),
    }


def inkling_cases():
    def case(text, finish='stop'):
        return dict(prompt=ASKED, tools=[LOOKUP], text=text, finish=finish)
    return {
        'answer': case(M + X + 'Hi 🙂' + E),
        'thought_answer': case(M + T + 'Add them.' + E + M + X + 'Four.' + E),
        'calls': case(M + T + 'Two.' + E + M + 'lookup' + J + '{"name":"lookup","args":{"ticket":413}}' + E + M + 'search' + J
            + '{"name": "search", "args": {"pattern": "é🙂\\n", "threshold": 0.5, "budget": 1e16, "z": null, "a": [1.0, true], "z": 2}}' + E),
        'unclosed_call': case(M + 'lookup' + J + '{"name":"lookup","args":{}}'),
        'truncated_call': case(M + X + 'Calling.' + E + M + 'lookup' + J + '{"name":"lookup","args":{"ticket":4', 'length'),
        'call_at_limit': case(M + 'lookup' + J + '{"name":"lookup","args":{}}', 'length'),
        'escapes': case(M + 'f' + J + '{"name":"f","args":{"s":"a\\u007fb\\tc","x":1.0000000000000003}}' + E),
        'truncated_answer': case(M + X + 'Four is <|', 'length'),
        'partial_marker': case(M + X + 'a <|'),
        'empty_call': case(M + 'lookup' + J + E),
        'between_messages': case(M + X + 'A' + E + 'stray' + M + X + 'B' + E),
        'unmarked': case('hello'),
        'bad_json': case(M + 'lookup' + J + '{"name":' + E),
        'no_args': case(M + 'lookup' + J + '{"name":"lookup"}' + E),
    }


def replayed(path, decode):
    served = json.loads((Path(__file__).parent / 'served-fixtures' / f'{path.name}.json').read_text())
    return {f'served_{i}': dict(prompt=decode(case['prompt_ids']), tools=case['tools'], output_ids=case['output_ids'], finish=case['finish']) for i, case in enumerate(served['cases'])}


def qwen(path):
    """mlx_lm.server's response loop, without the HTTP response around it."""
    from mlx_lm.generate import TextStateMachine, make_text_state_machine
    from mlx_lm.server import ToolCallFormatter
    from mlx_lm.tokenizer_utils import load
    tok = load(path)
    machine = make_text_state_machine(tok)
    found = dict(qwen_cases(), **replayed(path, tok.decode))
    for case in found.values():
        ids = case.pop('output_ids', None) or tok.encode(case['text'], add_special_tokens=False)
        finish = case['finish']
        if ids[-1] in tok.eos_token_ids:
            ids = ids[:-1]
        prompt = tok.encode(case['prompt'], add_special_tokens=False)
        state = 'reasoning' if tok.rfind_think_start(prompt) > tok.rfind_think_end(prompt) else 'normal'
        state = previous = tok.response_initial_state or state
        detokenizer = tok.detokenizer
        detokenizer.reset()
        pieces = []
        for token in ids:
            detokenizer.add_token(token)
            pieces.append(detokenizer.last_segment)
        detokenizer.finalize()
        last = detokenizer.last_segment
        if finish == 'length':
            last = pieces.pop() + last
        sm = machine.make_state(state)
        reasoning = text = tool_text = ''
        tool_texts = []
        made_tool_call = False
        for piece, reason in [(piece, None) for piece in pieces] + [(last, finish)]:
            if reason == 'stop':
                sm, state = TextStateMachine.discard(sm)
                segments = [(state, '')]
            elif reason == 'length':
                sm, segments = TextStateMachine.split(sm, piece)
                sm, flushed, state = TextStateMachine.flush(sm)
                segments.append((state, flushed))
            else:
                sm, segments = TextStateMachine.split(sm, piece)
            for state, clean in segments:
                if state == 'reasoning':
                    reasoning += clean
                elif state == 'tool':
                    tool_text += clean
                elif state == 'normal':
                    if previous == 'tool':
                        tool_texts.append(tool_text)
                        tool_text = ''
                        made_tool_call = True
                    text += clean
                previous = state
        if previous == 'tool' and tool_text:
            tool_texts.append(tool_text)
            made_tool_call = True
        calls = ToolCallFormatter(tok.tool_parser, case['tools'] or None)(tool_texts)
        case['text'] = tok.decode(ids)
        # The Rust decoder keeps what this detokenizer trims or holds back, so such a case would not compare.
        assert ''.join(pieces) + (last if finish == 'length' else '') == case['text'] and (finish == 'length' or not last), case['text']
        case.update(output_ids=ids, reasoning=reasoning, content=text, tool_calls=[call['function'] for call in calls],
            unparsed_calls=len(tool_texts) - len(calls), finish_reason='tool_calls' if finish == 'stop' and made_tool_call else finish)
    return found, ('mlx-lm', 'transformers', 'tokenizers')


def inkling(path):
    """The response loop of the hub's Inkling server, without the HTTP response around it."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    channels = {tok.convert_tokens_to_ids(tag): field for tag, field in (
        (T, 'reasoning_content'), (X, 'content'), (J, 'tool_calls'), (E, None), (M, None), ('<|content_model_end_sampling|>', None))}
    found = dict(inkling_cases(), **replayed(path, tok.decode))
    for case in found.values():
        ids = case.pop('output_ids', None) or tok.encode(case['text'], add_special_tokens=False)
        if ids[-1] == tok.eos_token_id:
            ids = ids[:-1]
        message = dict(content='')
        field, pending = None, []
        def flush(final):
            if not field or not pending:
                return
            value = tok.decode(pending, skip_special_tokens=False)
            if field == 'tool_calls':
                if not final:
                    return
                tool = json.loads(value)
                message.setdefault(field, []).append(dict(name=tool['name'], arguments=json.dumps(tool['args'])))
            else:
                if not final and value.endswith('\ufffd'):
                    return
                message[field] = message.get(field, '') + value
            pending.clear()
        case.update(text=tok.decode(ids, skip_special_tokens=False), output_ids=ids)
        try:
            for token in ids:
                if token in channels:
                    flush(True)
                    field = channels[token]
                    pending.clear()
                elif field:
                    pending.append(token)
                    if len(pending) >= 12 and field != 'tool_calls':
                        flush(False)
            cut = case['finish'] == 'length' and field == 'tool_calls'
            if not cut:
                flush(True)
        except (KeyError, TypeError, ValueError) as error:
            case['raises'] = type(error).__name__
            continue
        calls = message.get('tool_calls', [])
        case.update(reasoning=message.get('reasoning_content', ''), content=message['content'], tool_calls=calls, unparsed_calls=int(cut and bool(pending)),
            finish_reason='length' if case['finish'] == 'length' else 'tool_calls' if calls else 'stop')
    return found, ('transformers', 'tokenizers')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=('qwen', 'inkling'))
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    path = args.checkpoint
    found, packages = (qwen if args.family == 'qwen' else inkling)(path)
    files = [path / name for name in ('chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json')]
    print(json.dumps(dict(
        checkpoint=path.name,
        sha256={file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files if file.exists()},
        versions={p: metadata.version(p) for p in packages},
        cases=found,
    ), ensure_ascii=False))


if __name__ == '__main__':
    main()
