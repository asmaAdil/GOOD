from pathlib import Path
from typing import Any, Dict, Union

import yaml


class ConfManager:
    _conf_params: Dict[str, Any] = {}

    def __init__(self) -> None:
        raise TypeError("You can't initialize instance of ConfManager class")

    @classmethod
    def init(cls, conf_file_path: str) -> None:
        cls._conf_params = cls._parse_by_file(conf_file_path)
        print("ConfManager init done\r\n")
        print(ConfManager._conf_params)

    @classmethod
    def get(cls, param_name: str) -> Any:
        """

        :param param_name:
        :return:
        """

        return cls._conf_params[param_name]

    @classmethod
    def has_param(cls, param_name: str) -> bool:
        try:
            cls.get(param_name)
        except KeyError:
            return False
        return True

    @classmethod
    def update_by_dict(cls, params: Dict[str, Any]) -> None:
        for key in params:
            cls._conf_params[key] = params[key]
        print("ConfManager updated")
        print(params)

    @classmethod
    def update_by_file(cls, new_conf_file: str) -> None:
        temp_conf_params = cls._parse_by_file(new_conf_file)
        cls.update_by_dict(temp_conf_params)

    @classmethod
    def get_all_params(cls) -> Dict[str, Any]:
        """Get all the configuration parameters as dict. It returns the copy of all parameters, so you can change it
        without affecting all params in ConfManager.
        :return: A dict containing all parameters.
        """
        return cls._conf_params.copy()

    @staticmethod
    def _parse_by_file(yml_file: Union[str, Path]) -> Dict[str, Any]:
        # Load the configuration file
        with Path(yml_file).open(mode="r") as stream:  # pylint: disable=unspecified-encoding
            dict_yaml: Dict[str, Any] = yaml.safe_load(stream)
            return dict_yaml
