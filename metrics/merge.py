from typing import List, Tuple

class Merge(object):
    def __init__(self, pair: Tuple[str, str], freq: int):
        self.pair = pair
        self.freq = freq
    
    @classmethod
    def parse_file_entry(cls, line: str) -> "Merge":
        try:
            spl = line.split(" ")
            return cls((spl[0], spl[1]), int(spl[2]))
        except (IndexError, TypeError) as err:
            raise ValueError(f"Invalid merge entry format: {line}", err)
            
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        return f'{self.pair}: {self.freq}'
    

def read_merges(file: str) -> List[Merge]:
    res = []
    with open(file, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            res.append(Merge.parse_file_entry(line))
    return res