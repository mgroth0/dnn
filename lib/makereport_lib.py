# **** upload URLs will need to not be static since apparently wolfram can't really handle uploads to the same url over and over. but redirects can be used, and make different kinds of cloud objects too

from wolframclient.language import wl

from lib.defaults import *
from lib.wolf.wolfpy import WOLFRAM

PUBLIC_REPORT_URL = 'https://www.wolframcloud.com/obj/9e1d2a8d-660a-4119-a31c-7ebacab6ae57'

def update_report(new_url):
    co = WOLFRAM.cloud_deploy(wl.HTTPRedirect(new_url), url=PUBLIC_REPORT_URL, public=True)
    log(f'finished deploying redirect:{co[0]}')

@log_invokation(with_args=True)
def upload_webpage(htmlDoc, wolfFolder, permissions='Private',resource_folder=None):
    with Temp('temp.html', w=htmlDoc.getCode()) as t:
        co = WOLFRAM.copy_file(t, f'{wolfFolder}/index.html', permissions=permissions)
    with Temp('temp.css', w=htmlDoc.stylesheet) as t:
        WOLFRAM.copy_file(t, f'{wolfFolder}/style.css')
    if resource_folder is not None:
        WOLFRAM.copy_file(resource_folder, f'Resources/{wolfFolder}')
    return co[0]
