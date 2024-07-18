from OmniEvent.arguments import DataArguments, ModelArguments, TrainingArguments, ArgumentParser
from OmniEvent.input_engineering.seq2seq_processor import type_start, type_end
from OmniEvent.backbone.backbone import get_backbone
from OmniEvent.model.model import get_model
from OmniEvent.input_engineering.seq2seq_processor import EDSeq2SeqProcessor
from OmniEvent.evaluation.metric import compute_seq_F1
from OmniEvent.trainer_seq2seq import Seq2SeqTrainer

def main():
    parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file="config/all-datasets/ed/s2s/duee.yaml")

    training_args.output_dir = 'output/DuEE1.0/ED/seq2seq/t5-base/'
    data_args.markers = ["<event>", "</event>", type_start, type_end]

    backbone, tokenizer, config = get_backbone(model_type=model_args.model_type, 
                                        model_name_or_path=model_args.model_name_or_path, 
                                        tokenizer_name=model_args.model_name_or_path, 
                                        markers=data_args.markers,
                                        new_tokens=data_args.markers)
    model = get_model(model_args, backbone)

    train_dataset = EDSeq2SeqProcessor(data_args, tokenizer, data_args.train_file)
    eval_dataset = EDSeq2SeqProcessor(data_args, tokenizer, data_args.validation_file)
    metric_fn = compute_seq_F1

    trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=metric_fn,
            data_collator=train_dataset.collate_fn,
            tokenizer=tokenizer,
        )
    trainer.train()


if __name__ == "__main__":
     main()