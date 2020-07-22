from wolframclient.language import wl, wlexpr

from mlib.file import Folder
from mlib.term import log_invokation
from mlib.web.api import API
from mlib.web.html import arg_tags
from mlib.wolf.wolf_lang import CloudObject, Function, If
from mlib.wolf.wolfpy import WolframService, mwl

class ExperimentDataBinAPI(API):
    def __init__(self, parentFolder, data_database, id_database, *args, dev=False,**kwargs):
        dev_s = '_dev' if dev else ''
        apiFile = Folder(parentFolder)[f'databin{dev_s}.wl']

        self.data_database = data_database
        self.id_database = id_database

        super().__init__(apiFile, *args, **kwargs)



    @log_invokation()
    def build_api_fun(self, serv: WolframService):
        serv._.exprs += [wlexpr('CloudSymbol["APILog"] = "started experiment_databin api"')]
        serv.dataUser = serv.fullMessage["IDdata"]
        serv.dataDATA = serv.fullMessage["DATAdata"]
        serv.sessionID = wl.ToString(serv.fullMessage["sessionID"])

        serv.coID = CloudObject(self.id_database.file.abspath)
        serv.coDATA = CloudObject(self.data_database.file.abspath)

        serv.existingAccounts = mwl.load_json(serv.coID)
        serv.accountData = mwl.load_json(serv.coDATA)

        serv.thisAccount = wl.Select(
            wl.Values(serv.existingAccounts),
            Function(
                wlexpr("xxx"),
                wl.Equal(wlexpr("xxx"), serv.dataUser)
            )
        )

        @self.service.if_
        def _(blk, _condition=wl.Equal(wl.Length(serv.thisAccount), 1)):
            blk.id = wl.Part(wl.Select(
                wl.Keys(blk.existingAccounts),
                Function(
                    wlexpr("xxx"),
                    wl.Equal(blk.existingAccounts[wlexpr("xxx")], blk.dataUser)
                )
            ), 1)  # wolfram indices
        @self.service.else_
        def _(blk):
            blk.AllIDs = wl.Keys(serv.existingAccounts)
            blk.id = wl.ToString(If(
                wl.Not(wl.SameQ(wl.Length(blk.AllIDs), 0)),
                wl.Plus(wl.Max(wl.Map(wlexpr('ToExpression'), blk.AllIDs)), 1),
                1
            ))
            blk.existingAccounts[blk.id] = blk.dataUser
            blk.save_json(blk.coID, blk.existingAccounts)
            blk.accountData[blk.id] = wl.Association()
            blk.save_json(blk.coDATA, blk.accountData)

        serv.accountData[serv.id][serv.sessionID] = serv.dataDATA
        serv.save_json(serv.coDATA, serv.accountData)



    def apiElements(self): return super().apiElements().extended(*arg_tags(

    ).objs)

    @classmethod
    def cs(cls):
        super().cs()
        return '''
        sessionID = new Date().getTime()
        class ExperimentDataBinAPI
            constructor: (@api_url) ->
                @ip = myIP()
            prep: (IDdata,DATAdata,sync=true) ->
                message = {IDdata,DATAdata,sessionID}
                message = JSON.stringify message
                messages = message.match(/.{1,5500}/g)
                i = 1 # wolfram index
                messageID = @ip + "***" + new Date().getTime().toString()
                [messages,i,messageID]
            push: (IDdata,DATAdata) -> 
                [messages, i, messageID] = @prep(IDdata,DATAdata)
                for m in messages
                    url = QueryURL(
                          @api_url, #inner.API_URL
                          {
                            message: m
                            index:  i
                            total: messages.length
                            messageID
                          }
                        )
                    log("pushing data(#{i}) with url: #{url}")
                    wcGET url
                    i += 1 
            push_async: (IDdata,DATAdata,onFinish) ->
                [messages, i, messageID] = @prep(IDdata,DATAdata)
                urls = []
                responses = []
                for m in messages
                    url = QueryURL(
                          @api_url, #inner.API_URL
                          {
                            message: m
                            index:  i
                            total: messages.length
                            messageID
                          }
                        )
                    urls.push(url)
                    i+=1
                recurse_push_async = (ii) ->
                    log("pushing data(#{ii}) with url: #{urls[ii]}")
                    if ii is (messages.length-1)
                        wcGET_async(urls[ii], (t)->
                                    onFinish(t)
                                )
                    else
                        wcGET_async(urls[ii], (t)->
                                recurse_push_async(ii+1)
                            )
                recurse_push_async(0)
    '''
