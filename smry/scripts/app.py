from smry.app.base import PageComponent, ListComponent
from smry.app.header import HeaderComponent
from smry.app.texts import SummaryTitle
from smry.app.load import FileLoadComponent
from smry.app.session import StateInitializer
from smry.app.summary import SummarySelector, SummarizerComponent, SummaryGenerator

def main() -> int:
    (
            PageComponent()
            .set_elements(
                header=ListComponent().set_components([
                    HeaderComponent(),
                    StateInitializer(),
                    ]),
                sidebar=ListComponent().set_components([
                    FileLoadComponent(),
                    SummarySelector()
                    ]),
                page=ListComponent().set_components([
                    SummaryTitle(),
                    SummarizerComponent(),
                    SummaryGenerator()
                    ])
                )
            .call()
            )
    return 0

if __name__ == "__main__":
    exit(main())
