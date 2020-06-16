from copy import deepcopy

from wolframclient.language import wlexpr, wl

from mlib.boot.mlog import log
from mlib.boot.mutil import enum, listkeys, listfilt
from lib.web import HTMLVar
from lib.wolf.wolfpy import WOLFRAM, weval
class APIDict:
    def __init__(self, symbolName):
        csb = weval(wl.ToString(wlexpr("$CloudSymbolBase[[1]]"))) + '/'

        self.symbolName = symbolName

        COname = csb + symbolName
        self.COname = COname
        log(f'self.COname:{COname}')

        # will loose data
        _HARD_RESET = False

        r = WOLFRAM.cloud_get(COname)

        backup = weval(wlexpr(
            'CopyFile[CloudObject["' + COname + '"],CloudObject["Backups/"<>ToString[UnixTime[]]]]'
        ))[0]
        log(f'backuped API to {backup}')

        failed = str(r) in ["$Failed", 'None']
        if _HARD_RESET or failed:
            log('APIDict ' + symbolName + ' does not exist. creating.')
            WOLFRAM.cloud_put({}, COname)

        self.getKey = "GETTHISVARIABLEABCDEFG"
        self.var1 = "namespace"
        self.var2 = "setValue"

        self.apiURL = WOLFRAM.cloud_deploy(wlexpr(
            'APIFunction[{"' + self.var1 + '" -> "String","' + self.var2 + '''" -> String}, (
            If[#''' + self.var2 + '''=="''' + self.getKey + '''",
            data=CloudSymbol["''' + self.symbolName + '''"];
            data[[Sequence@@StringSplit[#''' + self.var1 + ''',"."]]]
            ,
            data=CloudSymbol["''' + self.symbolName + '''"];
            data[[Sequence@@StringSplit[#''' + self.var1 + ',"."]]] = #' + self.var2 + ''';
            CloudSymbol["''' + self.symbolName + '''"] = data;
            data=CloudSymbol["''' + self.symbolName + '''"];
            data[[Sequence@@StringSplit[#''' + self.var1 + ''',"."]]]
            ]
            )&]'''
        ))[0]

        self.unusedKeys = listkeys(weval(wlexpr(
            f'CloudSymbol["{self.symbolName}"]'
        )))

    def __setitem__(self, names, value):
        assert isinstance(value, str) or (isinstance(value, dict) and len(list(value.keys())) == 0)
        weval(wlexpr(
            f'data=CloudSymbol["{self.symbolName}"]'
        ))
        setCom = 'data'
        for n in names:
            setCom += f'[["{n}"]]'
        if isinstance(value, str):
            setCom += '=\"' + value + '\"'
        else:
            setCom += '=Association[]'
        weval(wlexpr(setCom))
        return weval(wlexpr(
            f'CloudSymbol["{self.symbolName}"]=data'
        ))

    def __delitem__(self, key):
        return weval(wlexpr(
            f'CloudSymbol["{self.symbolName}"]=KeyDrop[CloudSymbol["{self.symbolName}"],{{"{key}"}}]'
        ))

    def __getitem__(self, item):
        def tryIt(self, data):
            namesCompleted = []
            success = False
            for idx, i in enum(item):
                try:
                    data = data[i]
                except KeyError:
                    toMake = deepcopy(namesCompleted)
                    toMake.append(i)
                    if idx == len(item) - 1:
                        self[toMake] = ""
                    else:
                        self[toMake] = {}
                    break
                namesCompleted.append(i)
                if len(namesCompleted) == len(item):
                    success = True
            return data, success
        success = False
        while not success:
            data = weval(wlexpr(
                f'CloudSymbol["{self.symbolName}"]'
            ))

            rData, success = tryIt(self, data)

        if isinstance(item, tuple):
            item = item[0]
        self.unusedKeys = listfilt(
            lambda e: e != item,
            self.unusedKeys
        )

        return rData

    def apiElements(self):
        return [
            HTMLVar("API_URL", self.apiURL),
            HTMLVar("API_GET_KEY", self.getKey),
            HTMLVar("API_VAR1", self.var1),
            HTMLVar("API_VAR2", self.var2)
        ]

    @staticmethod
    def js():
        return '''
function simpleGET(url,callback) {
    
    let flag = true
    var t = ''
    while (flag) {
        let Http = new XMLHttpRequest()
        Http.open("GET", url,false) // 3rd param blocks
        Http.send()
        t = Http.responseText
        flag = t.includes("Unable to acquire kernel")
    }
    if (typeof callback !== 'undefined') {
        callback(t)
    }
    
    //if (typeof callback !== 'undefined') {
      //  Http.onreadystatechange = (_) => { 
        //    callback(Http.responseText)
        //}
    //}
}
function apiFun(namespace,value,callback) {
    let url=baseURL + "?"+apiVar1+"="+namespace+"&"+apiVar2+"="+value
    simpleGET(url,callback)
}
apiGET =(ns,cb) => apiFun(ns,getKey,cb)
deQuote = (str) => str.substring(1,str.length-1)
forEachByTag = (tag,cb) => Array.from(document.getElementsByTagName(tag)).forEach(cb)
apiVariable =(key) => document.getElementById(key).innerHTML
window.onload = function() {
    baseURL = apiVariable("API_URL")
    getKey = apiVariable("API_GET_KEY")
    apiVar1 = apiVariable("API_VAR1")
    apiVar2 = apiVariable("API_VAR2")
    forEachByTag("textarea",function(textarea){
        apiGET(textarea.id, function(response) {
            textarea.value = deQuote(response)
            textarea.addEventListener("input", function () {
                apiFun(textarea.id,textarea.value)
            })
        })
    })
    forEachByTag("p",function(para){
        apiGET(para.id,function(response) {
            para.innerHTML = deQuote(response)
        })
    })
}
'''
