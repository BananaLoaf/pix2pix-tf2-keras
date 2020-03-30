from etc.config import Config

if __name__ == '__main__':
    config = Config.from_cli()

    from pix2pix import Pix2Pix
    gan = Pix2Pix.new_run(config)
    gan.train()
