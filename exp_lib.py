from abc import ABC, abstractmethod
from dataclasses import dataclass

import inspect

from experiment_databin_api import ExperimentDataBinAPI
from mlib.boot.bootutil import pwd
from mlib.boot.mutil import isitr, gen_password
from mlib.boot.stream import listitems
from mlib.file import Folder, File
from mlib.term import log_invokation
from mlib.web.database import Database
from mlib.web.js import JS, compile_coffeescript
from mlib.web.makereport_lib import write_webpage
from mlib.web.simple_admin_api import SimpleAdminAPI
from mlib.web.web import arg_tags, HTMLPage, JScript

class OnlineHumanExperiment(ABC):
    def __init__(self, RESOURCES_ROOT: Folder, _DEV: bool = None):
        assert _DEV is not None
        self._DEV = _DEV
        self.RESOURCES_ROOT = RESOURCES_ROOT
        self.RESOURCES_ROOT = Folder(self.RESOURCES_ROOT)
        self.EXP_FOLDER = File(inspect.getfile(self.__class__)).parent
        self.FIG_FOLDER = Folder(self.EXP_FOLDER['figs'])
        self.changelist = self.EXP_FOLDER['changelist.yml']
        self.VERSIONS = self.changelist
        self.THIS_VERSION = listitems(self.VERSIONS)[-1]
        self.ROOT = self.EXP_FOLDER['build/site']

    def setup_databases_and_apis(self):
        if self._DEV:
            self.DATABASE_IDS = self._setup_database_and_api('ids_dev', hidden=True)
            self.DATABASE_DATA = self._setup_database_and_api('data_dev', hidden=False)
        else:
            self.DATABASE_IDS = self._setup_database_and_api('ids', hidden=True)
            self.DATABASE_DATA = self._setup_database_and_api('data', hidden=False)
        self.EXP_API = ExperimentDataBinAPI(
            self.EXP_FOLDER,
            self.DATABASE_DATA, self.DATABASE_IDS
        )

    def retrieve_api_info(self, name, hidden):
        database_folder = {
            False: self.EXP_FOLDER,
            True : self.EXP_FOLDER.hidden_version(pwd())
        }[hidden]
        database_file = database_folder[f'{name}.json']
        password_file = database_file.parent[f'._{database_file.name_pre_ext}_password.txt']
        return database_file, password_file

    def _setup_database_and_api(self, name, hidden):
        database_file, password_file = self.retrieve_api_info(name, hidden)
        database = Database(database_file)
        password = password_file.read() if password_file.exists else gen_password()
        password_file.write(password)
        SimpleAdminAPI(database, allow_set=False, password=password)
        return database

    @abstractmethod
    def pre_build(self): pass

    @log_invokation(with_result=True)
    def build(
            self,
            _UPLOAD_RESOURCES,
            _LOCAL_ONLY
    ):
        _DEV = self._DEV
        if not _LOCAL_ONLY:
            input(f'make sure to change version ( {self.changelist.abspath} )')
            input(f'make sure to test all supported oss and browsers')
        self.pre_build()
        self.setup_databases_and_apis()
        html = self.html_body_children()
        if not isitr(html): html = [html]
        htmlDoc = HTMLPage(
            'index',
            self.EXP_API.apiElements(),
            JScript(JS(compile_coffeescript(self.EXP_API.cs()), onload=False)),
            JScript(JS(self.EXP_FOLDER[f'{self.EXP_FOLDER.name}.coffee'])),
            *html,
            arg_tags(
                VERSION=self.THIS_VERSION[0],
                RESOURCES_ROOT=self.RESOURCES_ROOT.wcurl,
                RESOURCES_ROOT_REL=self.RESOURCES_ROOT.rel_to(self.ROOT),
                IS_DEV=_DEV
            ),
            style=self.css()
        )
        write_webpage(
            htmlDoc=htmlDoc,
            root=self.ROOT,
            resource_root_file=self.RESOURCES_ROOT,
            upload_resources=_UPLOAD_RESOURCES,
            WOLFRAM=not _LOCAL_ONLY,
            DEV=_DEV
        )

    @abstractmethod
    def css(self): pass
    @abstractmethod
    def html_body_children(self): pass

    @abstractmethod
    def analyze(self): pass
