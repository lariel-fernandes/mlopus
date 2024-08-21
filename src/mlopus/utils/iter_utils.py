import collections.abc
from typing import Generic, Callable, TypeVar, Iterable, Any, List

T = TypeVar("T")

R = TypeVar("R")  # Type of result
Token = Any | None  # Type of token


class Page(Generic[R]):
    """Page container to be used with Paginator."""

    def __init__(self, token: Token, results: List[R]):
        self.token = token
        self.results = results

    def map_results(self, mapper: Callable[[R], T]) -> "Page":
        """Get same page with each result mapped by callback."""
        return Page[T](self.token, [mapper(x) for x in self.results])

    def map_page(self, mapper: Callable[[R], T]) -> "Page":
        """Get same page with all results mapped by callback."""
        return Page[T](self.token, mapper(self.results))


class Paginator(Generic[R], collections.abc.Iterable[List[R]]):
    """Functional results paginator.

    Example:
        page_size, max_results = 3, 9

        def get_page(token):
            data = range(token := token or 0, new_token := token + page_size)
            return Page(new_token if new_token < max_results else None, data)

        for x in Paginator(get_page).map_with(lambda x: x + 1):
            print(x)  # 1, 2, 3, 4... 9
    """

    def __init__(self, get_page: Callable[[Token], Page[R]]):
        self._get_page = get_page

    def __iter__(self) -> Iterable[List[str]]:  # noqa
        token = None

        while True:
            yield (page := self._get_page(token)).results

            if not (token := page.token):
                break

    def map_results(self, mapper: Callable[[R], T]) -> "Paginator":
        """Get same paginator with each result mapped by callback."""
        return Paginator[T](lambda token: self._get_page(token).map_results(mapper))

    def map_pages(self, mapper: Callable[[List[R]], List[T]]) -> "Paginator":
        """Get same paginator with each page mapped by callback."""
        return Paginator[T](lambda token: self._get_page(token).map_page(mapper))

    def collapse(self) -> "Paginator":
        """Get a paginator with all possible results in a single page."""
        return Paginator[R](lambda _: Page(None, list(result for page in self for result in page)))

    @classmethod
    def single_page(cls, page_data: Iterable[R]) -> "Paginator":
        """Get a paginator of a single page from a results iterable."""
        return Paginator[R](lambda token: Page(token=None, results=list(page_data)))
