from abc import ABC, abstractmethod

from loggy import log
from mutil import File, isstr
DARK_CSS = '''
body {
    background:black;
    color: white;
}
'''


class HTMLDoc:
    def __init__(self,
                 *children,
                 # stylesheet=DARK_CSS.name,
                 stylesheet=DARK_CSS,
                 js=''
                 ):
        self.children = children
        self.stylesheet = stylesheet
        self.js = js
    def getCode(self):
        log('getting HTML code for HTML doc')
        ml = '<!DOCTYPE html>'
        ml += '<html>'
        ml += '<head>'
        ml += '<link rel="stylesheet" href="'
        ml += 'style.css'
        # ml += self.stylesheet
        ml += '">'
        ml += """
        
        <script>
        """
        # probably from syntax highlight injection
        ml += self.js.replace('\u200b', '')
        ml += """
        </script>
        
        """
        ml += '</head>'
        ml += HTMLBody(*self.children).getCode()
        ml += '</html>'
        return ml

class HTMLObject(ABC):
    def __init__(self, style='', id=None):
        self.style = style
        self.id = id
    @staticmethod
    @abstractmethod
    def tag():
        pass

    @abstractmethod
    def contents(self):
        pass

    @abstractmethod
    def attributes(self):
        pass

    def _attributes(self):
        atts = self.attributes()
        if self.id is not None:
            atts = atts + ' id="' + self.id + '"'
        if atts:
            return ' ' + atts
        else:
            return ''

    @staticmethod
    @abstractmethod
    def closingTag():
        pass

    def _style(self):
        if self.style:
            return ' style="' + self.style + '"'
        else:
            return ''

    def getCode(
            self):
        return '<' + self.tag() + self._style() + self._attributes() + '>' + self.contents() + self.closingTag()

class HTMLContainer(HTMLObject):
    def __init__(self, *args, **kwargs):
        super(HTMLContainer, self).__init__(**kwargs)
        self.objs = args

    def closingTag(self):
        return '</' + self.tag() + '>'

    @staticmethod
    @abstractmethod
    def sep():
        pass

    def contents(self):
        ml = ''
        for o in self.objs:
            if isstr(o):
                ml += o
            else:
                ml += o.getCode()
            if isstr(o) or 'hidden' not in o.attributes():
                ml += self.sep()
        return ml

class HTMLVar(HTMLContainer):
    def __init__(self, id, var):
        super(HTMLVar, self).__init__(var, id=id)
    def attributes(self): return 'hidden'
    @staticmethod
    def tag(): return 'p'
    @staticmethod
    def sep(): return ''

class HTMLBody(HTMLContainer):
    def attributes(self): return ''
    @staticmethod
    def tag(): return 'body'
    @staticmethod
    def sep(): return '<br>'



class Div(HTMLContainer):
    def attributes(self): return ''
    @staticmethod
    def tag(): return 'div'
    @staticmethod
    def sep(): return ''

class Span(HTMLContainer):
    def attributes(self): return ''
    @staticmethod
    def tag(): return 'span'
    @staticmethod
    def sep(): return ''

class Table(HTMLContainer):
    def attributes(self): return ''
    @staticmethod
    def tag(): return 'table'
    @staticmethod
    def sep(): return ''

class TableRow(HTMLContainer):
    def attributes(self): return ''
    @staticmethod
    def tag(): return 'tr'
    @staticmethod
    def sep(): return ''

class DataCell(HTMLContainer):
    def attributes(self): return ''
    @staticmethod
    def tag(): return 'td'
    @staticmethod
    def sep(): return ''

class P(HTMLContainer):
    def attributes(self): return ''
    @staticmethod
    def tag(): return 'p'
    @staticmethod
    def sep(): return ''

class TextArea(HTMLContainer):
    def __init__(self, text='', **kwargs):
        super(TextArea, self).__init__(text, **kwargs)
        self.text = text
    def attributes(self): return ''
    @staticmethod
    def tag(): return 'textarea'
    @staticmethod
    def sep(): return ''

class Hyperlink(HTMLContainer):
    def __init__(self, label, url):
        super(Hyperlink, self).__init__(label)
        self.url = url
    @staticmethod
    def tag(): return 'a'
    @staticmethod
    def sep(): return ''
    def attributes(self): return f'href="{self.url}"'

class HTMLChild(HTMLObject, ABC):
    def contents(self): return ''
    def closingTag(self): return ''

class _Br(HTMLChild):
    @staticmethod
    def tag():
        return 'br'
    def attributes(self):
        return ''
Br = _Br()

class HTMLImage(HTMLChild):
    def __init__(self, url, *args, **kwargs):
        super(HTMLImage, self).__init__(*args, **kwargs)
        self.url = url
    @staticmethod
    def tag(): return 'img'
    def attributes(self): return f'src="{self.url}" alt="an image" width="500"'
