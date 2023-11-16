
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
      "cad_df": {"ref": "download_cad"},
    },
    "prepared_cad": {
      "type": "prepare_cad",
      "flat_cad_df": {"ref": "flattened_cad"},
      "tokenizer": tokenizer
    },
    "finetuned": {
      "type": "finetune",
      "model": {
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
        },
      },
      "data": {"ref": "prepared_cad"},
      "tokenizer": tokenizer,
    },
  }
}
