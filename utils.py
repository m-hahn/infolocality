import csv
import random
import operator
import itertools
import functools
from collections import defaultdict
from typing import *

import numpy as np
import pandas as pd

DELIMITER = '#'
EPSILON = 10 ** -8

flat = itertools.chain.from_iterable

T = TypeVar("T", bound=Any)

def shuffled(xs: Iterable[T]) -> List[T]:
    xs = list(xs)
    random.shuffle(xs)
    return xs

def first(xs: Iterable[T]) -> T:
    return next(iter(xs))

def is_monotonically_increasing(xs: np.ndarray) -> bool:
    return (np.diff(xs) > -EPSILON).all()

def the_only(xs):
    """ Return the single value of a one-element iterable """
    x, = xs
    return x

def write_dicts(file, lines):
    lines_iter = iter(lines)
    first_line = next(lines_iter)
    writer = csv.DictWriter(file, first_line.keys())
    writer.writeheader()
    writer.writerow(first_line)
    for line in lines_iter:
        writer.writerow(line)        

def write_dfs(file, dfs):
    def gen():
        for df in dfs:
            for _, row in df.iterrows():
                yield dict(row)
    write_dicts(file, gen())


class Delimiter:
    def parts(self, x: T) -> Iterable[Union[str, T]]:
        raise NotImplementedError

    def delimit(self, x: Sequence) -> Sequence:
        return restorer(x)(flat(self.parts(x)))

    def delimit_string(self, x: str) -> str:
        return "".join(self.parts(x))

    def delimit_sequence(self, x: Sequence[T]) -> Tuple[Union[str, T]]:
        return tuple(flat(self.parts(x)))

    def delimit_array(self, x: pd.Series) -> pd.Series:
        return functools.reduce(operator.add, self.parts(x))

class LeftDelimiter(Delimiter):
    def parts(self, x: T) -> List[Union[str, T]]:
        return [DELIMITER, x]

class RightDelimiter(Delimiter):
    def parts(self, x: T) -> List[Union[str, T]]:    
        return [x, DELIMITER]

class BothDelimiter(Delimiter):
    def parts(self, x: T) -> List[Union[str, T]]:        
        return [DELIMITER, x, DELIMITER]

class NullDelimiter(Delimiter):
    def parts(self, x: T) -> List[T]:
        return [x]

def restorer(x: Sequence) -> Callable[[Iterable], Sequence]:
    if isinstance(x, str):
        return "".join
    else:
        return type(x)

def strip(xs: Sequence[T], y: T) -> Sequence[T]:
    """ Like str.strip but for any sequence. """
    result = xs
    if xs[0] == y:
        result = result[1:]
    if result[-1] == y:
        result = result[:-1]
    return result

def sequence_transformer(f):
    """ Return f' which applies f to a sequence preserving type. """
    def wrapped(s, *a, **k):
        restore = restorer(s)
        result = f(s, *a, **k)
        return restore(result)
    return wrapped

def delimited_sequence_transformer(f):
    """ Return f' which applies f to a sequence preserving type and delimitation. """
    def wrapped(s, *a, **k):
        restore = restorer(s)
        l = list(s)
        has_left_delimiter = l[0] == DELIMITER
        has_right_delimiter = l[-1] == DELIMITER
        l2 = list(strip(s, DELIMITER))        
        r = f(l2, *a, **k)
        if has_left_delimiter:
            r = itertools.chain([DELIMITER], r)
        if has_right_delimiter:
            r = itertools.chain(r, [DELIMITER])
        return restore(r)
    return wrapped

def test_delimited_sequence_transformer():
    one = "abc"
    two = "def#"
    three = "#ghi"
    four = "#jkl#"
    
    f = delimited_sequence_transformer(lambda x: (x[1], x[2], x[0]))
    assert f(one) == "bca"
    assert f(two) == "efd#"
    assert f(three) == "#hig"
    assert f(four) == "#klj#"
    assert f(list(one)) == list("bca")
    assert f(list(two)) == list("efd#")
    assert f(list(three)) == list("#hig")
    assert f(list(four)) == list("#klj#")
    assert f(tuple(one)) == tuple("bca")
    assert f(tuple(two)) == tuple("efd#")
    assert f(tuple(three)) == tuple("#hig")
    assert f(tuple(four)) == tuple("#klj#")    

def test_delimiters():
    r = RightDelimiter()
    l = LeftDelimiter()
    b = BothDelimiter()
    n = NullDelimiter()

    assert r.delimit("abc") == "abc#"
    assert l.delimit("abc") == "#abc"
    assert b.delimit("abc") == "#abc#"
    assert n.delimit("abc") == "abc"

    assert r.delimit(tuple("abc")) == tuple("abc#")
    assert l.delimit(tuple("abc")) == tuple("#abc")
    assert b.delimit(tuple("abc")) == tuple("#abc#")
    assert n.delimit(tuple("abc")) == tuple("abc")

    assert r.delimit(list("abc")) == list("abc#")
    assert l.delimit(list("abc")) == list("#abc")
    assert b.delimit(list("abc")) == list("#abc#")
    assert n.delimit(list("abc")) == list("abc")

    assert r.delimit_string("abc") == "abc#"
    assert l.delimit_string("abc") == "#abc"
    assert b.delimit_string("abc") == "#abc#"
    assert n.delimit_string("abc") == "abc"

    assert r.delimit_sequence(tuple("abc")) == tuple("abc#")
    assert l.delimit_sequence(tuple("abc")) == tuple("#abc")
    assert b.delimit_sequence(tuple("abc")) == tuple("#abc#")
    assert n.delimit_sequence(tuple("abc")) == tuple("abc")

    assert r.delimit_sequence(list("abc")) == tuple("abc#")
    assert l.delimit_sequence(list("abc")) == tuple("#abc")
    assert b.delimit_sequence(list("abc")) == tuple("#abc#")
    assert n.delimit_sequence(list("abc")) == tuple("abc")

    import pandas as pd
    strings = pd.Series(["abc"]*10000)
    assert (r.delimit_array(strings) == pd.Series(["abc#"]*10000)).all()
    assert (l.delimit_array(strings) == pd.Series(["#abc"]*10000)).all()
    assert (b.delimit_array(strings) == pd.Series(["#abc#"]*10000)).all()
    assert (n.delimit_array(strings) == strings).all()

if __name__ == '__main__':
    import nose
    nose.runmodule()
