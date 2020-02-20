import click
import yaml


def parse_yaml(cfg):
    return yaml.load(cfg, yaml.FullLoader)


@click.command()
@click.option('--model', help='Choose model from inception, resnet or mobilenet')
@click.argument('config', type=click.File('r'), required=True)
def cli(model, config):

    config = parse_yaml(config.read())
