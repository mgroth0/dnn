_current_step_index = null
_current_step = null
_steps = null
class Step
    constructor: (@save_after, @controls) ->
        @readyForKey = false
        @finish_unixms = null
        @finished_data_push = false
        @last_step = null
        @is_last_trial = false
    run        : ->
        log "running #{@}"
    onkey      : (e) ->
    data       : ->
        d =
            finish_unixms: @finish_unixms
    finish     : ->
        @finish_unixms = new Date().getTime()
        if _current_step_index < _steps.length - 1
            id.prog.value += 1
            id.top.style["justify-content"] = "center" #so load indicator is centered
            run_next = =>
#                if @ instanceof
                if @is_last_trial
                    push_data_async((t)=>
                        @finished_data_push = true
                    )
                    wait_timeout_recurse = =>
                        if not @finished_data_push
                            log('waiting for data to push...')
                            #                            id.loadIndicator.alternate(['/', '\\'], 1000)
                            setTimeout(=>
                                wait_timeout_recurse()
                            , 100)
                        else
                            log('fading last trial loading indicator')
                            id.loadIndicator.fade(->
                                log('faded last trial loading indicator')
                                #                               id.loadIndicator.stop_alternating()
                                id.top.style["justify-content"] = null
                                run_step(_current_step_index + 1)
                            )
                    #
                    id.loadIndicator.unfade()
                    wait_timeout_recurse()
                else
                    if @save_after
                        push_data_async((t)=>
                            @finished_data_push = true
                        )
                    else
                        @finished_data_push = true
                    id.loadIndicator.fade(->
#                        id.loadIndicator.stop_alternating()
                        id.top.style["justify-content"] = null
                        run_step(_current_step_index + 1)
                    )

            if @last_step?
                id.loadIndicator.unfade()
                wait_timeout_recurse = =>
                    if not @last_step.finished_data_push
                        log('waiting for data to push...')
                        #                        id.loadIndicator.alternate(['/', '\\'], 1000)
                        setTimeout(=>
                            wait_timeout_recurse()
                        , 100)
                    else
                        run_next()
                wait_timeout_recurse()
            else
                run_next()
class Instruction extends Step
    constructor: (@text, @extra_id) -> super(false, [SPACE_BAR])
    run        : ->
        super()
        id.top.style['justify-content'] = 'center'
        if !@extra_id?
#            apparently id isn't even needed since ids are auto-added to Window???????
            id['text-container'].style.height = '100%' # so the text is centered
            id.text.style.height = '100%' # so the text is centered
        else
            id['text-container'].style.height = null
            id.text.style.height = null
        id['text-container'].unfade()
        id.text.style['text-align'] = 'left' #best alignment for typing
        if @extra_id?
            @extra_id.appear(false, true)
            @extra_id.op = 0
        id.text.type(@text.replace('TOKEN_VERSION', id.VERSION.innerHTML), =>
            if @extra_id?
                @extra_id.unfade(=>
                    @readyForKey = true
                )
            else
                @readyForKey = true
        )

    finish: ->
        @extra_id?.fade()
        id['text-container'].fade(=>
            id.text.style['text-align'] = 'center' #for trials
            super()
        )
user = null
class Login extends Instruction
    constructor: (text) ->
        super(text, id.login)
        @controls = []
        @save_after = true

        #        FOR SAFARI
        #        if (id.birthday.type!="date"){ #if browser doesn't support input type="date", load files for jQuery UI Date Picker
        #            document.write('<link href="http://ajax.googleapis.com/ajax/libs/jqueryui/1.12.1/themes/base/jquery-ui.css" rel="stylesheet" type="text/css" />\n')
        #            document.write('<script src="http://ajax.googleapis.com/ajax/libs/jqueryui/1.12.1/jquery-ui.min.js"><\/script>\n')
        #        }
        #        if (id.birthday.type != "date") #if browser doesn't support input type="date", initialize date picker widget:
        #            $(document).ready(->

        #        nah, I want all browsers to act the same so I'll just override any native implementations
        #        $('#birthday').datepicker()

        #            )

        change_days = () ->
            month = id.birthday_month.value
            year = id.birthday_year.value
            monthNum = new Date(Date.parse(month + " 1," + year)).getMonth() + 1
            num_days = new Date(year, monthNum, 0).getDate()
            current_day = id.birthday_day.value
            while id.birthday_day.firstChild
                id.birthday_day.removeChild(id.birthday_day.lastChild)
            for d in [0..num_days]
                node = document.createElement("OPTION")
                node.innerHTML = d.toString()
                id.birthday_day.appendChild(node)
            if Number(current_day) in [0..num_days]
                id.birthday_day.value = current_day
        id.birthday_month.onchange = ->
            change_days()
        id.birthday_year.onchange = ->
            change_days()

        id.login_button.onclick = (e) =>
            user =
                first_name: id.first_name.value.toUpperCase()
                last_name : id.last_name.value.toUpperCase()
                birthday  : "#{id.birthday_month.value}-#{id.birthday_day.value}-#{id.birthday_year.value}"
            @.finish()
    finish     : ->
        openFullscreen(tag.body)
        super()


left_control = LEFT_ARROW
right_control = RIGHT_ARROW
choice_text_center = '<-?->'

class BinaryChoice_Base extends Step
    constructor     : (@im, @t, @left_choice, @right_choice) ->
        super(false, [left_control, right_control])
        @choice = null
        @fix_path = true
        @presentation_time_unixms = null
        @reaction_time_ms = null
    run             : ->
        super()
        RESOURCES_ROOT_REL = id.RESOURCES_ROOT_REL.innerHTML
        id.top.style["justify-content"] = "center" # so the text and img is centered
        id['text-container'].disappear() # so the img is centered
        #        id.text.innerHTML = center_str("#{@left_choice} <-?-> #{@right_choice}",'?')
        id.text.innerHTML = center_str("#{@left_choice} #{choice_text_center} #{@right_choice}", '?')
        id['im-container'].appear()
        id.cross.appear()
        id.im.src = if @fix_path then path_join(RESOURCES_ROOT_REL, @im) else @im
        @display_stimulus()
    display_stimulus: -> throw 'abstract'
    onkey           : (e) ->
        @reaction_time_ms = new Date().getTime() - @presentation_time_unixms
        @choice = (if (e.keyCode == left_control.keyCode) then @left_choice else @right_choice)
    data            : ->
        merge({@im, @t, @choice, @reaction_time_ms}, super())
    finish          : ->
        id.im.disappear()
        id['im-container'].disappear()
        id['text-container'].disappear()
        #        id['text-container'].style.height = null
        #        id.text.style.height = null
        id.top.style["justify-content"] = null
        super()



class BinaryChoice_Flash extends BinaryChoice_Base
    display_stimulus: ->
        setTimeout(=>
            id.cross.disappear()
            setTimeout(=>
                id.im.disappear()
                id['text-container'].style.height = '100%' # so the text is centered
                id.text.style.height = '100%' # so the text is centered
                id['text-container'].appear()
                @readyForKey = true
            , @t)
            id.im.appear()
            @presentation_time_unixms = new Date().getTime()
        , 1000)

class BinaryChoice_Wait extends BinaryChoice_Base
    display_stimulus: ->
        setTimeout(=>
            id.cross.disappear()
            #            setTimeout(=>
            #                id.im.disappear()
            #                id.text.innerHTML = "#{@left_choice}<- ->#{@right_choice}"
            #                id['text-container'].style.height = '100%'
            #                id.text.style.height = '100%'
            #                id['text-container'].appear()
            #                @readyForKey = true
            #            , @t)
            #            id.top.style['justify-content'] = 'center' # so the text is above img
            id['text-container'].style.height = '10%' # so the text is above img
            id.text.style.height = '10%' # so the text is above img
            id['text-container'].appear()
            id.im.appear()
            @presentation_time_unixms = new Date().getTime()
            @readyForKey = true
        , 1000)


class Thank_You extends Instruction
    constructor: (text) ->
        super(text, id.feedback)
        @controls = []

#        dont need these any more since I prevented scrolling in the first place
#addEventListener("keydown", (e) ->
# prevent space and arrow keys from scrolling
#    if [32, 37, 38, 39, 40].indexOf(e.keyCode) > -1
#        e.preventDefault()
#, false)

keyup (e) ->
    if _current_step? and e.keyCode in _current_step.controls.map((c)->c.keyCode)
        if _current_step.readyForKey
            _current_step.readyForKey = false
            _current_step.onkey(e)
            _current_step.finish()
run_step = (idx) ->
    id['text-container'].fade()
    _current_step_index = idx
    _current_step = _steps[_current_step_index]
    _current_step.run()
check_platform = (run_after) ->
    browser = platform.name
    os = platform.os.family
    accepted_browsers = ['Firefox', 'Safari', 'Chrome']
    accepted_oss = ['OS X']
    if browser in accepted_browsers and os in accepted_oss
        run_after()
    else
        id.text.innerHTML = "Sorry, this experiment has not yet successfully passed tests on your platform (you are using #{os} and #{browser}). Please access this site using a supported operating system and browser: (Operating Systems: [#{accepted_oss.join(',')}], Browsers: [#{accepted_browsers.join(',')}])."
        id['top'].unfade()
        id['text-container'].unfade()
        id['text'].unfade()
run_steps = (steps) ->
    for i in [1..(steps.length - 1)]
        steps[i].last_step = steps[i - 1]
        if steps[i] instanceof Thank_You
            steps[i - 1].is_last_trial = true
    _steps = steps
    id.prog.max = _steps.length - 1
    id.top.unfade()
    run_step(0)
push_data = ->
    data =
        version : id.VERSION.innerHTML
        steps   : _steps.map((s)->s.data())
        feedback: id.textarea.value
    new ExperimentDataBinAPI(inner.API_URL).push(
            user,
            data
    )
push_data_async = (onFinish) ->
    data =
        version : id.VERSION.innerHTML
        steps   : _steps.map((s)->s.data())
        feedback: id.textarea.value
    new ExperimentDataBinAPI(inner.API_URL).push_async(
            user,
            data,
            onFinish
    )

id.button.addEventListener(
        "click",
    ->
        id.feedback_status.innerHTML = 'Please wait for data to upload...'
        push_data_async(->
            id.feedback_status.innerHTML = 'Feedback submitted, thank you!'
        )
)

fsHandler = ->
    if not window.IS_DEV_MODE
        if (document.webkitIsFullScreen or document.mozFullScreen or (document.msFullscreenElement is not null))
            id.top.appear()
            id['progress-container'].appear()
            id.fullscreen_button.disappear()
        else
            id.top.disappear()
            id['progress-container'].disappear()
            id.fullscreen_button.appear()

id['fullscreen_button'].onclick = (e) ->
    openFullscreen(tag.body)

if document.addEventListener
    document.addEventListener('fullscreenchange', fsHandler, false)
    document.addEventListener('mozfullscreenchange', fsHandler, false)
    document.addEventListener('MSFullscreenChange', fsHandler, false)
    document.addEventListener('webkitfullscreenchange', fsHandler, false)

