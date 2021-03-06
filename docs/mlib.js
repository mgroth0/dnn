// Generated by CoffeeScript 2.5.1			//GEN
var C_KEY, DOWN_ARROW, E, GET, GET_async, Key, LEFT_ARROW, N_KEY, QueryURL, RIGHT_ARROW, SPACE_BAR, S_KEY, UP_ARROW, _parse_ip, appendToBody, autoYRange, bool, center_str, extend, forEachByTag, id, inner, just_email_href_stuff, keydown, keyup, log, log_invokation, merge, myIP, myIP_async, onload_funs, openFullscreen, openSocketClient, openTab, path_join, retic, tag, tap, tic, wcGET, wcGET_async;			//GEN
			//GEN
onload_funs = [];			//GEN
			//GEN
window.onload = function() {			//GEN
  return onload_funs.map(function(f) {			//GEN
    return f();			//GEN
  });			//GEN
};			//GEN
			//GEN
id = new Proxy({}, {			//GEN
  get: function(target, name) {			//GEN
    return $('#' + name)[0];			//GEN
  }			//GEN
});			//GEN
			//GEN
tag = new Proxy({}, {			//GEN
  get: function(target, name) {			//GEN
    return $(name)[0];			//GEN
  }			//GEN
});			//GEN
			//GEN
//noinspection JSUnusedGlobalSymbols			//GEN
inner = new Proxy({}, {			//GEN
  get: function(target, name) {			//GEN
    return id[name].innerHTML;			//GEN
  }			//GEN
});			//GEN
			//GEN
GET = function(url) {			//GEN
  var Http;			//GEN
  Http = new XMLHttpRequest();			//GEN
  Http.open("GET", url, false); // 3rd param blocks, but this is deprecated			//GEN
  Http.send();			//GEN
  return Http.responseText;			//GEN
};			//GEN
			//GEN
GET_async = function(url, onResponse) {			//GEN
  var Http;			//GEN
  Http = new XMLHttpRequest();			//GEN
  Http.open("GET", url);			//GEN
  Http.send();			//GEN
  return Http.onreadystatechange = function() {			//GEN
    if (Http.readyState === 4) {			//GEN
      if (Http.status === 200) {			//GEN
        return onResponse(Http.responseText);			//GEN
      } else {			//GEN
        return alert(`bad HTTP request: url=${url},status=${Http.status},response=${Http.responseText}`);			//GEN
      }			//GEN
    }			//GEN
  };			//GEN
};			//GEN
			//GEN
wcGET = function(url) {			//GEN
  var t;			//GEN
  t = "Unable to acquire kernel";			//GEN
  while (t.includes("Unable to acquire kernel")) {			//GEN
    t = GET(url);			//GEN
  }			//GEN
  return t;			//GEN
};			//GEN
			//GEN
wcGET_async = function(url, onResponse) {			//GEN
  var t;			//GEN
  t = "Unable to acquire kernel";			//GEN
  return GET_async(url, function(t) {			//GEN
    if (t.includes("Unable to acquire kernel")) {			//GEN
      return wcGET_async(url, onResponse);			//GEN
    } else {			//GEN
      return onResponse(t);			//GEN
    }			//GEN
  });			//GEN
};			//GEN
			//GEN
Object.defineProperty(String.prototype, "de_quote", {			//GEN
  value: function de_quote() {			//GEN
        return this.substring(1, this.length - 1)			//GEN
    },			//GEN
  writable: true			//GEN
});			//GEN
			//GEN
//noinspection JSUnusedGlobalSymbols			//GEN
forEachByTag = (tag, cb) => {			//GEN
  return Array.from(document.getElementsByTagName(tag)).forEach(cb);			//GEN
};			//GEN
			//GEN
QueryURL = function(url, args_dict) {			//GEN
  var j, k, keys, len;			//GEN
  keys = Object.keys(args_dict);			//GEN
  if (keys.length) {			//GEN
    url = `${url}?`;			//GEN
  }			//GEN
  for (j = 0, len = keys.length; j < len; j++) {			//GEN
    k = keys[j];			//GEN
    url = `${url}${encodeURI(k)}=${encodeURI(args_dict[k])}&`;			//GEN
  }			//GEN
  if (keys.length > 1) {			//GEN
    url = url.substring(0, url.length - 1);			//GEN
  }			//GEN
  return url;			//GEN
};			//GEN
			//GEN
Object.defineProperty(Element.prototype, "setText", {			//GEN
  value: function(t) {			//GEN
    switch (this.tagName) {			//GEN
      case "P":			//GEN
        return this.innerHTML = t;			//GEN
      case "TEXTAREA":			//GEN
        return this.value = t;			//GEN
      default:			//GEN
        throw `setText not coded for ${this.tagName}`;			//GEN
    }			//GEN
  },			//GEN
  writable: true			//GEN
});			//GEN
			//GEN
Object.defineProperty(Element.prototype, "disappear", {			//GEN
  value: function(just_hide) {			//GEN
    if ((typeof just_display === "undefined" || just_display === null) || !just_display) {			//GEN
      if (this.fade_timer != null) {			//GEN
        clearInterval(this.fade_timer);			//GEN
        this.fade_timer = null;			//GEN
      }			//GEN
      if (this.unfadetimer != null) {			//GEN
        clearInterval(this.unfadetimer);			//GEN
        this.unfadetimer = null;			//GEN
      }			//GEN
    }			//GEN
    this.op = 0;			//GEN
    if (this.style.display !== 'none') {			//GEN
      this._default_display = getComputedStyle(this).display;			//GEN
    }			//GEN
    if ((just_hide != null) && just_hide) {			//GEN
      return this.style.visibility = 'hidden';			//GEN
    } else {			//GEN
      return this.style.display = 'none';			//GEN
    }			//GEN
  },			//GEN
  writable: true			//GEN
});			//GEN
			//GEN
Object.defineProperty(Element.prototype, "setFadeInterval", {			//GEN
  value: function(interval) {			//GEN
    return this.fadeInterval = interval;			//GEN
  },			//GEN
  writable: true			//GEN
});			//GEN
			//GEN
Object.defineProperty(Element.prototype, "fade", {			//GEN
  value: function(onFinish, just_hide) {			//GEN
    if (this.fadeInterval == null) {			//GEN
      this.fadeInterval = 50;			//GEN
    }			//GEN
    if ((this.fade_timer == null) && this.style.display !== 'none' && this.style.visibility !== 'hidden') {			//GEN
      if (this.unfadetimer != null) {			//GEN
        clearInterval(this.unfadetimer);			//GEN
        this.unfadetimer = null;			//GEN
      } else {			//GEN
        this.op = 1; // initial opacity			//GEN
      }			//GEN
      return this.fade_timer = setInterval(() => {			//GEN
        if (this.op <= 0.01) {			//GEN
          this.op = 0;			//GEN
          clearInterval(this.fade_timer);			//GEN
          if ((this.just_hide != null) && just_hide) {			//GEN
            this.style.visibility = 'hidden';			//GEN
          } else {			//GEN
            this.disappear();			//GEN
          }			//GEN
          this.fade_timer = null;			//GEN
          if (onFinish != null) {			//GEN
            onFinish();			//GEN
          }			//GEN
        }			//GEN
        return this.op -= Math.max(this.op * 0.1, 0);			//GEN
      }, this.fadeInterval);			//GEN
    } else {			//GEN
      if (onFinish != null) {			//GEN
        return onFinish();			//GEN
      }			//GEN
    }			//GEN
  },			//GEN
  writable: true			//GEN
});			//GEN
			//GEN
Object.defineProperty(Element.prototype, "appear", {			//GEN
  value: function(just_display, but_hidden) {			//GEN
    if ((just_display == null) || !just_display) {			//GEN
      if (this.fade_timer != null) {			//GEN
        clearInterval(this.fade_timer);			//GEN
        this.fade_timer = null;			//GEN
      }			//GEN
      if (this.unfadetimer != null) {			//GEN
        clearInterval(this.unfadetimer);			//GEN
        this.unfadetimer = null;			//GEN
      }			//GEN
      this.op = 1;			//GEN
    }			//GEN
    if ((but_hidden != null) && but_hidden) {			//GEN
      this.style.visibility = 'hidden';			//GEN
    }			//GEN
    if (this.style.display === 'none') {			//GEN
      if (this.hasAttribute('data-display')) {			//GEN
        return this.style.display = this.getAttribute('data-display');			//GEN
      } else if (this._default_display) {			//GEN
        return this.style.display = this._default_display;			//GEN
      } else {			//GEN
        return this.style.display = 'block';			//GEN
      }			//GEN
    }			//GEN
  },			//GEN
  writable: true			//GEN
});			//GEN
			//GEN
Object.defineProperty(Element.prototype, "unfade", {			//GEN
  value: function(onFinish) {			//GEN
    if (this.fadeInterval == null) {			//GEN
      this.fadeInterval = 50;			//GEN
    }			//GEN
    if ((this.unfadetimer == null) && ((this.fade_timer != null) || this.style.display === 'none' || this.style.visibility === 'hidden')) {			//GEN
      if (this.fade_timer != null) {			//GEN
        clearInterval(this.fade_timer);			//GEN
        this.fade_timer = null;			//GEN
      } else {			//GEN
        this.op = 0; // initial opacity			//GEN
      }			//GEN
      this.appear(true);			//GEN
      if (this.style.visibility === 'hidden') {			//GEN
        this.style.visibility = 'visible';			//GEN
      }			//GEN
      return this.unfadetimer = setInterval(() => {			//GEN
        if (this.op >= 1) {			//GEN
          clearInterval(this.unfadetimer);			//GEN
          this.unfadetimer = null;			//GEN
          if (onFinish != null) {			//GEN
            onFinish();			//GEN
          }			//GEN
        }			//GEN
        return this.op += Math.max(Math.min(this.op * 0.1, 1), 0.01);			//GEN
      }, this.fadeInterval);			//GEN
    } else {			//GEN
      if (onFinish != null) {			//GEN
        return onFinish();			//GEN
      }			//GEN
    }			//GEN
  },			//GEN
  writable: true			//GEN
});			//GEN
			//GEN
Object.defineProperty(Element.prototype, "op", {			//GEN
  get: function() {			//GEN
    return Number(this.style.opacity);			//GEN
  },			//GEN
  set: function(op) {			//GEN
    this.style.opacity = op;			//GEN
    return this.style.filter = 'alpha(opacity=' + op * 100 + ")";			//GEN
  }			//GEN
});			//GEN
			//GEN
// writable: true			//GEN
Object.defineProperty(Element.prototype, "alternate", {			//GEN
  value: function(htmls, period) {			//GEN
    var alt_recurse, i;			//GEN
    this._stop_alternating = false;			//GEN
    i = 0;			//GEN
    alt_recurse = () => {			//GEN
      this.innerHTML = htmls[i];			//GEN
      if (i === (htmls.length - 1)) {			//GEN
        i = 0;			//GEN
      } else {			//GEN
        i++;			//GEN
      }			//GEN
      return setTimeout(() => {			//GEN
        if (!this._stop_alternating) {			//GEN
          return alt_recurse();			//GEN
        }			//GEN
      }, period);			//GEN
    };			//GEN
    return alt_recurse();			//GEN
  },			//GEN
  writable: true			//GEN
});			//GEN
			//GEN
Object.defineProperty(Element.prototype, "stop_alternating", {			//GEN
  value: function() {			//GEN
    return this._stop_alternating = true;			//GEN
  },			//GEN
  writable: true			//GEN
});			//GEN
			//GEN
Object.defineProperty(Element.prototype, "type", {			//GEN
  value: function(ss, onFinish) {			//GEN
    var i, j, ref, s;			//GEN
    s = ss.charAt(0);			//GEN
    for (i = j = 1, ref = ss.length - 1; (1 <= ref ? j <= ref : j >= ref); i = 1 <= ref ? ++j : --j) {			//GEN
      if (ss.charAt(i) === ' ') {			//GEN
        s = s + ' ';			//GEN
      } else {			//GEN
        s = s + '&nbsp';			//GEN
      }			//GEN
    }			//GEN
    i = 0;			//GEN
    return this.type_timer = setInterval(() => {			//GEN
      this.innerHTML = s;			//GEN
      if (i === ss.length - 1) {			//GEN
        clearInterval(this.type_timer);			//GEN
        if (onFinish != null) {			//GEN
          onFinish();			//GEN
        }			//GEN
      }			//GEN
      i += 1;			//GEN
      if (s.substr(i, 5) === '&nbsp') {			//GEN
        return s = s.substring(0, i) + ss.charAt(i) + s.substring(i + 5);			//GEN
      } else {			//GEN
        return s = s.substring(0, i) + ss.charAt(i) + s.substring(i + 1);			//GEN
      }			//GEN
    }, 20);			//GEN
  },			//GEN
  writable: true			//GEN
});			//GEN
			//GEN
path_join = function(...args) {			//GEN
  return args.map(function(part, i) {			//GEN
    if (i === 0) {			//GEN
      return part.trim().replace(/[\/]*$/g, '');			//GEN
    } else {			//GEN
      return part.trim().replace(/(^[\/]*|[\/]*$)/g, '');			//GEN
    }			//GEN
  }).filter(function(x) {			//GEN
    return x.length;			//GEN
  }).join('/');			//GEN
};			//GEN
			//GEN
tic = Date.now();			//GEN
			//GEN
retic = function() {			//GEN
  return tic = Date.now();			//GEN
};			//GEN
			//GEN
log = function(s) {			//GEN
  var toc;			//GEN
  toc = Date.now();			//GEN
  return console.log(`[${(toc - tic) / 1000}]${s}`);			//GEN
};			//GEN
			//GEN
bool = function(s) {			//GEN
  return JSON.parse(s.toLowerCase());			//GEN
};			//GEN
			//GEN
keydown = function(f) {			//GEN
  return $(document).keydown(function(e) {			//GEN
    //        log("#{e.key} down")			//GEN
    return f(e);			//GEN
  });			//GEN
};			//GEN
			//GEN
keyup = function(f) {			//GEN
  return $(document).keyup(function(e) {			//GEN
    //        log("#{e.key} up")			//GEN
    return f(e);			//GEN
  });			//GEN
};			//GEN
			//GEN
Key = function(keyCode, str) {			//GEN
  return {keyCode, str};			//GEN
};			//GEN
			//GEN
SPACE_BAR = Key(32, "Space Bar");			//GEN
			//GEN
RIGHT_ARROW = Key(39, "Right Arrow Key");			//GEN
			//GEN
LEFT_ARROW = Key(37, "Left Arrow Key");			//GEN
			//GEN
UP_ARROW = Key(38, "Right Arrow Key");			//GEN
			//GEN
DOWN_ARROW = Key(40, "Left Arrow Key");			//GEN
			//GEN
S_KEY = Key(83, "S Key");			//GEN
			//GEN
C_KEY = Key(67, "C Key");			//GEN
			//GEN
N_KEY = Key(78, "N Key");			//GEN
			//GEN
//noinspection JSUnusedGlobalSymbols			//GEN
just_email_href_stuff = function() {};			//GEN
			//GEN
//  id.link.href = "#{id.link.href}body=#{encodeURI(template)}"			//GEN
openFullscreen = function(DOM_e) {			//GEN
  if (DOM_e.requestFullscreen) {			//GEN
    return DOM_e.requestFullscreen();			//GEN
  } else if (DOM_e.mozRequestFullScreen) { ///* Firefox */			//GEN
    return DOM_e.mozRequestFullScreen();			//GEN
  } else if (DOM_e.webkitRequestFullscreen) { ///* Chrome, Safari and Opera */			//GEN
    return DOM_e.webkitRequestFullscreen();			//GEN
  } else if (DOM_e.msRequestFullscreen) { ///* IE/Edge */			//GEN
    return DOM_e.msRequestFullscreen();			//GEN
  }			//GEN
};			//GEN
			//GEN
tap = function(o, fn) {			//GEN
  fn(o);			//GEN
  return o;			//GEN
};			//GEN
			//GEN
merge = function(...xs) {			//GEN
  if ((xs != null ? xs.length : void 0) > 0) {			//GEN
    return tap({}, function(m) {			//GEN
      var j, k, len, results, v, x;			//GEN
      results = [];			//GEN
      for (j = 0, len = xs.length; j < len; j++) {			//GEN
        x = xs[j];			//GEN
        results.push((function() {			//GEN
          var results1;			//GEN
          results1 = [];			//GEN
          for (k in x) {			//GEN
            v = x[k];			//GEN
            results1.push(m[k] = v);			//GEN
          }			//GEN
          return results1;			//GEN
        })());			//GEN
      }			//GEN
      return results;			//GEN
    });			//GEN
  }			//GEN
};			//GEN
			//GEN
//ipapi doesnt allow CORS			//GEN
//myIP = ->			//GEN
//    JSON.parse(GET('https://ipapi.co/json/')).ip			//GEN
//myIP_async = (handler) ->			//GEN
//    GET_async('https://ipapi.co/json/',(t)->			//GEN
//        handler(JSON.parse(t).ip)			//GEN
//    )			//GEN
_parse_ip = function(raw) {			//GEN
  return raw.split('\n').filter(function(l) {			//GEN
    return l.startsWith('ip');			//GEN
  })[0].replace('ip=', '');			//GEN
};			//GEN
			//GEN
myIP = function() {			//GEN
  return _parse_ip(GET('https://www.cloudflare.com/cdn-cgi/trace'));			//GEN
};			//GEN
			//GEN
myIP_async = function(handler) {			//GEN
  return GET_async('https://www.cloudflare.com/cdn-cgi/trace', function(t) {			//GEN
    return handler(_parse_ip(t));			//GEN
  });			//GEN
};			//GEN
			//GEN
center_str = function(str, marker) {			//GEN
  var mi;			//GEN
  mi = str.indexOf(marker) + 1;			//GEN
  while (mi < (str.length / 2 + 0.5)) {			//GEN
    str = '&nbsp' + str;			//GEN
    mi = str.indexOf(marker) + 1;			//GEN
  }			//GEN
  while (mi > (str.length / 2 + 0.5)) {			//GEN
    str = str + '&nbsp';			//GEN
    mi = str.indexOf(marker) + 1;			//GEN
  }			//GEN
  if (str.length % 2 === 0) {			//GEN
    return str.replace(marker, '&nbsp');			//GEN
  } else {			//GEN
    return str.replace(marker, '');			//GEN
  }			//GEN
};			//GEN
			//GEN
appendToBody = function(e) {			//GEN
  return tag.body.appendChild(e);			//GEN
};			//GEN
			//GEN
E = new Proxy({}, {			//GEN
  get: function(target, name) {			//GEN
    return function(my_inner) {			//GEN
      var e;			//GEN
      e = document.createElement(name);			//GEN
      e.innerHTML = my_inner;			//GEN
      return e;			//GEN
    };			//GEN
  }			//GEN
});			//GEN
			//GEN
Object.defineProperty(String.prototype, "afterFirst", {			//GEN
  value: function afterFirst(c) {			//GEN
        return this.substring(this.indexOf(c) + 1)			//GEN
    },			//GEN
  writable: true			//GEN
});			//GEN
			//GEN
Object.defineProperty(String.prototype, "beforeFirst", {			//GEN
  value: function afterFirst(c) {			//GEN
        return this.substring(0, this.indexOf(c) + 1)			//GEN
    },			//GEN
  writable: true			//GEN
});			//GEN
			//GEN
openTab = function(evt, tabName) {			//GEN
  var c, content, j, len, len1, len2, link, n, p, ref, results, tabcontent, tablinks;			//GEN
  tabcontent = document.getElementsByClassName("tabcontent");			//GEN
  for (j = 0, len = tabcontent.length; j < len; j++) {			//GEN
    content = tabcontent[j];			//GEN
    content.setFadeInterval(5);			//GEN
    //        content.fade()			//GEN
    content.disappear();			//GEN
  }			//GEN
  tablinks = document.getElementsByClassName("tablinks");			//GEN
  for (n = 0, len1 = tablinks.length; n < len1; n++) {			//GEN
    link = tablinks[n];			//GEN
    link.className = link.className.replace(" active", "");			//GEN
  }			//GEN
  evt.currentTarget.className += " active";			//GEN
  //    id[tabName].unfade()			//GEN
  id[tabName].appear();			//GEN
  ref = id[tabName].children[0].children;			//GEN
  results = [];			//GEN
  for (p = 0, len2 = ref.length; p < len2; p++) {			//GEN
    c = ref[p];			//GEN
    if (c.children.length > 0 && c.children[0].className.includes('plotly-graph-div')) {			//GEN
      //            hopefully this will fix the weird bug that causes these plots to regularly hide thier lines			//GEN
      results.push(Plotly.relayout(c.children[0], {}));			//GEN
    } else {			//GEN
      results.push(void 0);			//GEN
    }			//GEN
  }			//GEN
  return results;			//GEN
};			//GEN
			//GEN
//            and c.children[0].className?			//GEN
//                'xaxis.autorange': true,			//GEN
//                'yaxis.autorange': true			//GEN
openSocketClient = function({url, onopen, onmessage, onclose}) {			//GEN
  var ws;			//GEN
  ws = new WebSocket(url);			//GEN
  ws.onopen = function() {			//GEN
    return onopen.call(ws);			//GEN
  };			//GEN
  ws.onmessage = onmessage;			//GEN
  ws.onclose = onclose;			//GEN
  return ws;			//GEN
};			//GEN
			//GEN
autoYRange = function(ar) {			//GEN
  var dif, mn, mx, ten;			//GEN
  mn = Math.min(ar);			//GEN
  mx = Math.max(ar);			//GEN
  dif = mx - mn;			//GEN
  ten = dif * 0.1;			//GEN
  mn = mn - ten;			//GEN
  mx = mx + ten;			//GEN
  return [mn, mx];			//GEN
};			//GEN
			//GEN
Object.defineProperty(String.prototype, "shorten", {			//GEN
  value: function shorten(maxlen) {			//GEN
        if (this.length <= maxlen) {return this} else {return this.slice(0, maxlen) + ' ... '}			//GEN
    },			//GEN
  writable: true			//GEN
});			//GEN
			//GEN
Object.defineProperty(Object.prototype, "def", {			//GEN
  value: function def(functions) {			//GEN
        for (fun_name in functions) {			//GEN
            Object.defineProperty(this, fun_name, {			//GEN
                value: functions[fun_name]			//GEN
            })			//GEN
        }			//GEN
        return this			//GEN
    }			//GEN
});			//GEN
			//GEN
log_invokation = function(f) {			//GEN
  var ff;			//GEN
  ff = function(...args) {			//GEN
    var r, s;			//GEN
    s = f.name + '()';			//GEN
    log(`Invoking ${s}...`);			//GEN
    r = f(...args);			//GEN
    log(`Finished ${s}!`);			//GEN
    return r;			//GEN
  };			//GEN
  ff.name = f.name;			//GEN
  return ff;			//GEN
};			//GEN
			//GEN
extend = function(object, properties) {			//GEN
  var key, val;			//GEN
  for (key in properties) {			//GEN
    val = properties[key];			//GEN
    object[key] = val;			//GEN
  }			//GEN
  return object;			//GEN
};			//GEN
			//GEN
merge = function(...objects) {			//GEN
  var j, len, o, r;			//GEN
  r = {};			//GEN
  for (j = 0, len = objects.length; j < len; j++) {			//GEN
    o = objects[j];			//GEN
    extend(r, o);			//GEN
  }			//GEN
  return r;			//GEN
};			//GEN
			//GEN
