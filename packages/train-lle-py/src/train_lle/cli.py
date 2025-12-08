import argparse
import json
import sys
from .engine import Model, Trainer
from .optimizer import SGD, Adam
from .loss import MSE, CrossEntropy
from .dataset import Dataset
from .export import save_model, load_model
from .models import MLP

def main():
    parser = argparse.ArgumentParser(prog='train-lle', description='Local Learning Engine CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # init command
    init_parser = subparsers.add_parser('init', help='Initialize a new project')

    # train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('config', help='config file or dataset path')

    # test command
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('model', help='model file')
    test_parser.add_argument('dataset', help='dataset file')

    # export command
    export_parser = subparsers.add_parser('export', help='Export the model')
    export_parser.add_argument('model', help='model file')
    export_parser.add_argument('output', help='output file')

    # stats command
    stats_parser = subparsers.add_parser('stats', help='Show model statistics')
    stats_parser.add_argument('model', help='model file')

    args = parser.parse_args()

    if args.command == 'init':
        config = {
            'model': 'mlp',
            'config': {'input': 4, 'layers': [8, 8], 'output': 1},
            'optimizer': 'adam',
            'lr': 0.01,
            'epochs': 50,
            'loss': 'mse',
            'dataset': 'data.csv'
        }
        with open('train-config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print('Created train-config.json')

    elif args.command == 'train':
        config_path = args.config
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {'model': 'mlp', 'config': {'input': 4, 'layers': [8, 8], 'output': 1}, 'optimizer': 'adam', 'lr': 0.01, 'epochs': 50, 'loss': 'mse', 'dataset': config_path}
        
        dataset = Dataset.from_csv(config['dataset'])
        # Build model
        if config['model'] == 'mlp':
            model = Model(MLP.build(config['config']))
        else:
            raise ValueError('Model not supported')
        
        optimizer = Adam(config['lr']) if config['optimizer'] == 'adam' else SGD(config['lr'])
        loss = MSE() if config['loss'] == 'mse' else CrossEntropy()
        trainer = Trainer(optimizer, loss, config['epochs'])
        trainer.fit(model, dataset.inputs, dataset.targets)
        # Save model
        save_model('model.lle', model)
        print('Training complete, model saved to model.lle')

    elif args.command == 'test':
        model_path = args.model
        dataset_path = args.dataset
        model = load_model(model_path)
        dataset = Dataset.from_csv(dataset_path)
        trainer = Trainer(SGD(0), MSE(), 0)  # dummy
        loss_val = trainer.evaluate(model, dataset.inputs, dataset.targets)
        print(f'Test loss: {loss_val}')

    elif args.command == 'export':
        model_path = args.model
        output_path = args.output
        model = load_model(model_path)
        save_model(output_path, model)
        print(f'Model exported to {output_path}')

    elif args.command == 'stats':
        model_path = args.model
        model = load_model(model_path)
        params = model.params()
        total = sum(len(p.data) for p in params)
        print(f'Total parameters: {total}')

    else:
        parser.print_help()

if __name__ == '__main__':
    main()