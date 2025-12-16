from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class Node:
    """Class representing a node of the graph object"""
    code: str
    level: str
    name: str
    description: str = ""
    includes: str = ""
    includes_also: str = ""
    excludes: str = ""
    implementation_rule: str = ""
    parent_code: Optional[str] = None
    children_codes: List[str] = field(default_factory=list)
    
    def to_json(self) -> dict:
        """Convert node to JSON-serializable dictionary"""
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "Node":
        """Create a Node instance from a dictionary"""
        # Filter only fields that exist in the dataclass
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)