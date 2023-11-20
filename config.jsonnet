
local pretrained_model = "TheBloke/Llama-2-7B-GPTQ";
#local pretrained_model = "TheBloke/Llama-2-13B-GPTQ";
#local pretrained_model = "TheBloke/Llama-2-70B-GPTQ";


local tokenizer = {
    pretrained_model_name_or_path: pretrained_model
};

{
  "steps": {
    "download_cad": {
      "type": "download_cad"
    },
    "flattened_cad": {
      "type": "flatten_cad",
      cad_df: {"ref": "download_cad"},
    },
    "prepared_cad": {
      "type": "prepare_cad",
      flat_cad_df: {"ref": "flattened_cad"},
      tokenizer: tokenizer,
      sample_max_len: 512,
      block_max_len: 512,
    },
    "finetuned": {
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
        peft_config: {
          "type": "lora-config",
          r: 16,
          lora_alpha: 32,
          lora_dropout: 0.1,
          bias: "none",
          target_modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
          task_type: "CAUSAL_LM",
        },
      },
      data: {"ref": "prepared_cad"},
      tokenizer: tokenizer,
      max_steps_train: 200,
      max_steps_eval: 500,
    },
  }
}
