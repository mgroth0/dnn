# **** upload URLs will need to not be static since apparently wolfram can't really handle uploads to the same url over and over. but redirects can be used, and make different kinds of cloud objects too

from wolframclient.language import wl

from lib.defaults import *
from lib.wolf.wolfpy import WOLFRAM
PUBLIC_URL = 'https://www.wolframcloud.com/obj/9e1d2a8d-660a-4119-a31c-7ebacab6ae57'
def update_report(new_url):
    co = WOLFRAM.cloud_deploy(wl.HTTPRedirect(new_url), url=PUBLIC_URL, public=True)
    log(f'finished deploying redirect:{co[0]}')

def upload_webpage(doc, folder, permissions='Private'):
    log(f'uploading webpage: {doc}')
    with Temp('temp.html') as t:
        t.write(doc.getCode())
        co = WOLFRAM.copy_file(t, folder + '/' + 'index.html', permissions=permissions)
    with Temp('temp.css') as t:
        t.write(doc.stylesheet)
        WOLFRAM.copy_file(doc.stylesheet, folder + '/style.css')
    return co[0]
