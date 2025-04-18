Todos
[ ] only when i pass verbose the logger info should print otherwise not
[ ] Implement concurrent tool execution when multiple tools are called and write clean code for it
[ ] Implement processing of multiple samples at a time in simulation.py
[ ] get the values in the csv in data/experiments/{model name}.csv. if a file exists append to it. use pandas and keep csv related util to simulation.py these field values should be there.

This is the sample output of run_experiment
{'experiment': ExperimentResponse(created_at=datetime.datetime(2025, 4, 17, 12, 57, 7, 894483, tzinfo=tzutc()), id='65de3e59-34e2-468a-99fa-b449883b8d26', name='gpt-4.1-nano-2025-04-14-banking-tool_coordination1-1744894619', project_id='b4bc5059-6793-4693-9e02-a472939a91c6', updated_at=datetime.datetime(2025, 4, 17, 12, 57, 7, 894484, tzinfo=tzutc()), created_by='cad8171d-1412-425e-a759-179dd4d8a17a', additional_properties={'task_type': 16, 'dataset': None, 'aggregate_metrics': {}, 'ranking_score': None, 'rank': None, 'winner': False}), 'link': '<https://console-galileo-v2-staging.gcp-dev.galileo.ai/project/b4bc5059-6793-4693-9e02-a472939a91c6/experiments/65de3e59-34e2-468a-99fa-b449883b8d26>', 'message': 'Experiment gpt-4.1-nano-2025-04-14-banking-tool_coordination1-1744894619 has completed and results are available at <https://console-galileo-v2-staging.gcp-dev.galileo.ai/project/b4bc5059-6793-4693-9e02-a472939a91c6/experiments/65de3e59-34e2-468a-99fa-b449883b8d26'}>
