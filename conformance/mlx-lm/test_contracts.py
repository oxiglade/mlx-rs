import copy
import json
import unittest

from generate_contracts import cli_contract, hub_contract, compare_cli, compare_hub, qualify, jsonl, serialized


class Contracts(unittest.TestCase):
    def test_capture_round_trip_preserves_jsonl_field_order(self):
        generation = cli_contract()['generation']
        capture = {'prompt': generation['argv'][4], 'prompt_token_ids': generation['prompt_token_ids'],
                   'responses': [{'token_id': 12, 'text': 'hello', 'finish_reason': 'length'}]}
        initial = cli_contract(capture)
        restored = cli_contract(json.loads(serialized(initial))['generation']['capture'])
        self.assertEqual(initial, restored)
        self.assertIn('"token_id":12,"text":"hello","finish_reason":"length"',
                      restored['generation']['expected']['jsonl']['stdout_utf8'])

    def test_online_failure_preserves_existing_receipt(self):
        for case in hub_contract()['cases']:
            if case['name'].startswith('online_failure_no_fallback_'):
                self.assertIsNotNone(case['setup']['receipt'])
                self.assertEqual(case['expected']['receipt'], case['setup']['receipt'])

    def test_mutation_classes(self):
        self.assertGreaterEqual(len(qualify(hub_contract(), cli_contract())), 20)

    def test_offline_transport_is_never_ignored(self):
        for case in hub_contract()['cases']:
            if case['request']['offline']:
                actual = copy.deepcopy(case['expected'])
                actual['transport_calls'] = [{'operation': 'construct_client'}]
                self.assertEqual(compare_hub(case['expected'], actual), 'hub_transport')

    def test_missing_capture_is_not_a_passing_golden(self):
        case = cli_contract()['generation']
        self.assertIsNone(case['expected'])
        with self.assertRaisesRegex(ValueError, 'capture'):
            compare_cli(case['expected'], {'exit_code': 0})

    def test_diagnostics_cannot_replace_jsonl(self):
        expected = cli_contract()['writer_cases'][0]['expected']
        actual = copy.deepcopy(expected)
        actual['stdout_utf8'] = 'progress\n' + actual['stdout_utf8']
        self.assertEqual(compare_cli(expected, actual), 'stdout')

    def test_zero_flags_have_no_dangling_argument(self):
        for case in cli_contract()['argument_failures'][:3]:
            flag = '--' + case['name'][5:]
            self.assertEqual(case['argv'][case['argv'].index(flag) + 1], '0')
            self.assertEqual(case['argv'].count(flag), 1)
            self.assertEqual(case['argv'][:4], ['generate', '--model', '{fixture}', '--prompt'])

    def test_actual_stdout_is_the_record_source(self):
        expected = cli_contract()['writer_cases'][0]['expected']
        actual = copy.deepcopy(expected)
        records = copy.deepcopy(expected['records'])
        records[-1]['token_id'] += 1
        actual['stdout_utf8'] = jsonl(records)
        self.assertEqual(compare_cli(expected, actual), 'sampled_id')

    def test_changed_receipt_and_absences_are_rejected(self):
        expected = hub_contract()['cases'][0]['expected']
        for key in ['receipt', 'recorded_absences']:
            actual = copy.deepcopy(expected)
            actual[key] = None
            self.assertEqual(compare_hub(expected, actual), 'hub_integrity')

    def test_unrelated_transport_failure_is_not_success(self):
        expected = hub_contract()['cases'][0]['expected']
        actual = copy.deepcopy(expected)
        actual['error'] = {'variant': 'HubError::Api'}
        self.assertEqual(compare_hub(expected, actual), 'error_class')


if __name__ == '__main__':
    unittest.main()
