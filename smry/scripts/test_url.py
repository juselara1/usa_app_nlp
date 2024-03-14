from smry.load import UrlConfig, UrlLoader

def main() -> int:
    # config = UrlConfig(url="https://docs.pydantic.dev/latest/concepts/validators/")
    config = UrlConfig(
        url="https://developer.mozilla.org/es/docs/Web/HTML#tutoriales_para_principiantes",
    )
    document = UrlLoader().set_config(config).load()
    print(document)
    return 0

if __name__ == "__main__":
    exit(main())
