{
  "dataset_reader": {
    "type": "vqa_rusv2",
    "image_dir": "/content/train2014",
    "feature_cache_dir": "/net/nfs2.allennlp/data/vision/vqa/balanced_real/feature_cache",
    "image_loader": "torch",
    "image_featurizer": "resnet_backbone",
    "region_detector": "faster_rcnn",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": "DeepPavlov/rubert-base-cased"
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": "DeepPavlov/rubert-base-cased"
      }
    },
    "image_processing_batch_size": 4,
    "answer_vocab": {
      "type": "from_files",
      "directory": "/content/drive/MyDrive/allennlp_ru/vilbert_vqa_rus.deeppavlov-rubert-base-cased.vocab.tar.gz"
    },
    "multiple_answers_per_question": true,
  },
  "validation_dataset_reader": {
    "type": "vqa_rusv2",
    "image_dir": "/content/train2014",
    "feature_cache_dir": "/net/nfs2.allennlp/data/vision/vqa/balanced_real/feature_cache",
    "image_loader": "torch",
    "image_featurizer": "resnet_backbone",
    "region_detector": "faster_rcnn",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": "DeepPavlov/rubert-base-cased"
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": "DeepPavlov/rubert-base-cased"
      }
    },
    "image_processing_batch_size": 4,
    "answer_vocab": null,
    "multiple_answers_per_question": true,
  },
  "vocabulary": {
      "type": "from_files",
      "directory": "/content/drive/MyDrive/allennlp_ru/vilbert_vqa_rus.deeppavlov-rubert-base-cased.vocab.tar.gz"
    },
  "train_data_path": ["balanced_real_train", "balanced_real_val[1000:]"],
  "validation_data_path": "balanced_real_val[:1000]",
  "model": {
    "type": "vqa_vilbert_from_huggingface",
    "model_name": "DeepPavlov/rubert-base-cased",
    "image_feature_dim": 1024,
    "image_hidden_size": 1024,
    "image_num_attention_heads": 8,
    "image_num_hidden_layers": 6,
    "combined_hidden_size": 1024,
    "combined_num_attention_heads": 8,
    "pooled_output_dim": 1024,
    "image_intermediate_size": 1024,
    "image_attention_dropout": 0.1,
    "image_hidden_dropout": 0.1,
    "image_biattention_id": [0, 1, 2, 3, 4, 5],
    "text_biattention_id": [6, 7, 8, 9, 10, 11],
    "text_fixed_layer": 0,
    "image_fixed_layer": 0,
    "fusion_method": "mul",
    "ignore_text": false, # debug setting
    "ignore_image": false, # debug setting
  },
  "data_loader": {
    "batch_size": 1,
    "shuffle": true,
    "max_instances_in_memory": 10240
  },
  "trainer": {
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 4e-5,
      "correct_bias": true,
      "weight_decay": 0.01,
      "parameter_groups": [
        [["^embeddings\\.", "^encoder.layers1\\.", "^t_pooler\\."], {"lr": 4e-6}]
      ],
    },
    "learning_rate_scheduler": {
      "type": "linear_with_warmup",
      "warmup_steps": 5000
    },
    "validation_metric": "+vqa_score",
    "patience": 5,
    "num_epochs": 40,
    "num_gradient_accumulation_steps": 1,
  },
  "random_seed": 876170670,
  "numpy_seed": 876170670,
  "pytorch_seed": 876170670,
}
