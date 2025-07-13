import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class GameInfo:
    """Represents a document in a game context, with an ID, description, and optional metadata.
    This class is used to encapsulate information about a document, such as its
    unique identifier, a description, and any additional metadata that may be
    relevant for the game or application.
    Attributes:
        id (str): Unique identifier for the document, generated as a UUID.
        title (str): Title of the game.
        developer (str): Name of the game developer.
        release_date (str): Release date of the game.
        platforms (list[str]): List of platforms the game is available on.
        genre (str): Genre of the game.
        description (str): Brief description of the game.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = field(default_factory=str)
    developer: str = field(default_factory=str)
    release_date: str = field(default_factory=str)
    platforms: str = field(default_factory=str)
    genre: str = field(default_factory=str)
    description: str = field(default_factory=str)

    def to_dict(self) -> dict[str, str]:
        """Convert the GameDocument to a dictionary format.

        This method extracts all attributes of the GameDocument into a dictionary,
        which can be useful for serialization or storage in a database.

        Returns:
            Dict[str, Any]: Dictionary containing the document's attributes.
        """
        return {
            "id": self.id,
            "title": self.title,
            "developer": self.developer,
            "release_date": self.release_date,
            "platforms": self.platforms,
            "genre": self.genre,
            "description": self.description
        }

@dataclass
class GameDocument:
    """Represents a document in a game context, with an ID, description, and optional metadata.

    This class is used to encapsulate information about a document, such as its
    unique identifier, a description, and any additional metadata that may be
    relevant for the game or application.

    Attributes:
        id (str): Unique identifier for the document, generated as a UUID.
        description (str): Brief description of the document.
        metadata (Dict[str, Any]): Optional metadata associated with the document.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = field(default_factory=str)
    metadata: Dict[str, Any] = None

    @classmethod
    def from_gameinfo(cls, game_info: GameInfo) -> 'GameDocument':
        """Create a GameDocument from a GameInfo instance.
        This method converts a GameInfo object into a GameDocument, extracting
        relevant attributes and formatting them appropriately for the document.
        Args:
            game_info (GameInfo): The GameInfo instance to convert.
        Returns:
            GameDocument: A new GameDocument instance populated with data from the GameInfo.
        """

        return GameDocument(
            id=game_info.id,
            content=f"{game_info.title}\n{game_info.developer}\n{game_info.release_date}\n{game_info.platforms}\n{game_info.genre}\n{game_info.description}",
            metadata={
                "title": game_info.title,
                "developer": game_info.developer,
                "release_date": game_info.release_date,
                "platforms": ",".join(game_info.platforms),
                "genre": game_info.genre
            }
        )
