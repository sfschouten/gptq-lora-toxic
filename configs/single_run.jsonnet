local finetune_steps_lib = import 'finetune_steps.jsonnet';

{
    'steps': finetune_steps_lib.steps[std.extVar('step')],
}
