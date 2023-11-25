# MODEL
#local pretrained_model = std.extVar('pretrained_model');
local pretrained_model = "TheBloke/Llama-2-7B-GPTQ";


local tokenizer = {
    pretrained_model_name_or_path: pretrained_model
};


# Hyperparameters
local nr_epochs = 20;
local max_train_steps = 150;
//local nr_epochs = 1;
//local max_train_steps = 5;


# PEFT
local peft_target_modules_options = {
#    currently adding the lm_head doesn't work with adalora, because of small issue
#    "full": ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head'],
    "no-lm-head": ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
};
local peft_configs = {
    ["lora_" + targets_key]: {
        "type": "lora-config",
        r: 16,                                      # rank of matrices added for fine-tuning
        lora_alpha: 32,                             # determines scale of adapter output (= lora_alpha / r)
        lora_dropout: 0.1,
        bias: "none",
        "target_modules": peft_target_modules_options[targets_key],
        task_type: "CAUSAL_LM",
    }
    for targets_key in std.objectFields(peft_target_modules_options)
} + {
    ["adalora_" + targets_key]: {
        "type": "adalora-config",
        init_r: 24,                                 # initial rank of incremental matrices
        target_r: 16,                               # target average rank of incremental matrices
        tinit: max_train_steps,                     # nr. of steps of initial warmup, before decreasing budget
        tfinal: nr_epochs * max_train_steps / 2,    # nr. of final steps after the budget reaching target
        deltaT: 1,                                  # nr. of steps between two budget allocations
        lora_alpha: 32,
        lora_dropout: 0.1,
        bias: "none",
        "target_modules": peft_target_modules_options[targets_key],
        task_type: "CAUSAL_LM",
    }
    for targets_key in std.objectFields(peft_target_modules_options)
};


{
	"steps": {
		[context + '_' + peft_key]: {
		    "cad": {
				"type": "download_cad"
		    },
			"flat_cad": {
				"type": "flatten_cad",
				cad_df: {"ref": "cad"},
			},
			"prepared_cad": {
				"type": "prepare_cad",
				flat_cad_df: {"ref": "flat_cad"},
				tokenizer: tokenizer,
				sample_max_len: 512,
				block_max_len: 8192,
//				block_max_len: 512,
				include_context: context == 'context',
			},
			"finetuned_model": {
				"type": "finetune",
				model: {
					"type": "peft::get_peft_model",
					base_model: {
						"type": "transformers::AutoModelForCausalLM::from_pretrained",
						pretrained_model_name_or_path: pretrained_model,
						quantization_config: {
							"type": "gptq-config",
							bits: 4,
							use_exllama: false,
						},
						device_map: {"": "cuda:0"},
					},
					"peft_config": peft_configs[peft_key],
				},
				data: {"ref": "prepared_cad"},
				tokenizer: tokenizer,
				num_epochs: nr_epochs,
				min_acc_samples: 8,                     # minimum nr. of samples to accumulate gradients for before descending
				epoch_train_steps: max_train_steps,     # nr. of gradient descent steps per epoch
				min_samples_eval: 750,                  # min nr. of samples to validate with
			},
			"test_results": {
			    "type": "test",
			    model: {"ref": "finetuned_model"},
			    tokenized_data: {"ref": "prepared_cad"},
			    original_data: {"ref": "flat_cad"},
			    tokenizer: tokenizer,
			}
		}
		for context in ["context", "no-context"]
		for peft_key in std.objectFields(peft_configs)
	}
}
