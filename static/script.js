$('.figure-class').hide();

var lastScrollTop = window.pageYOffset || document.documentElement.scrollTop;

window.onscrollend = function() {
  var st = window.pageYOffset || document.documentElement.scrollTop;

  if (st > lastScrollTop) {
    // Scrolling down
    window.scrollTo(0, document.documentElement.scrollHeight || document.body.scrollHeight);
  } else {
    // Scrolling up
    window.scrollTo(0, 0);
  }

  lastScrollTop = st <= 0 ? 0 : st; // For Mobile or negative scrolling
};

function loadBackgroundImages(numberOfImages, uuid) {
  var video_frame = document.getElementById('video-frame');
  for (var i = 1; i <= numberOfImages; i++) {
    video_frame.style.backgroundImage = "url('static/sessions/" + uuid + "/image-" + ('0000000' + i).slice(-7) + ".png')";
  }
}

function changeBackgroundImage(imageNumber, uuid) {
  var video_frame = document.getElementById('video-frame');
  video_frame.style.backgroundImage = "url('static/sessions/" + uuid + "/image-" + ('0000000' + imageNumber).slice(-7) + ".png')";
}

var emotions;
// fetch('static/sessions/user-0.0.0.1/classes/predicts')
//     .then(response => response.text())
//     .then(data => {
//         emotions = data.split('\n').map(e => e.trim()).filter(e => e);
//     })
//     .catch(error => console.error('Failed to load file:', error));


var uuid;
var num_frames;
var predicts;
var classes;

function changeClass(imageNumber, numberOfImages) {
  var figure_class = document.getElementById('figure-class');
  var image_class = document.getElementById('image-class');
  var class_caption = document.getElementById('class-caption');
  var bias = (imageNumber - 1) / (numberOfImages - 1);
  figure_class.style.transform = 'translate(calc(-20vw + 14px + ' + bias + ' * (40vw - 28px)), calc(50vw * 9 / 32 - 3vw))';
  var values = predicts[imageNumber - 1];
  var max_value = 0;
  var pred_class = 0;
  for (var i = 1; i <= values.length; i++) {
    $('#predicts-list li:nth-child(' + i + ')').css('border', '1px solid #222');
    $('#predicts-list li:nth-child(' + i + ')').css('background', '#333');
    $('#predicts-list li:nth-child(' + i + ')').css('color', 'white');
    $('#predicts-list li:nth-child(' + i + ') p:nth-of-type(2)').text(values[i - 1] + '%');
    if (values[i - 1] > max_value) {
      max_value = values[i - 1];
      pred_class = i - 1;
    }
    if (values[i - 1] > 50) {
      $('#predicts-list li:nth-child(' + i + ')').css('border', '1px solid #99FF99');
    }
    if (values[i - 1] > 90) {
      $('#predicts-list li:nth-child(' + i + ')').css('background', '#99FF99');
      $('#predicts-list li:nth-child(' + i + ')').css('color', '#222');
    }
  }
  var emotion = classes[pred_class];
  image_class.src = 'static/resources/emotions/' + emotion + '.png';
  image_class.alt = emotion;
  class_caption.innerHTML = emotion;
}

function updateVideoInfo(frame) {
  if (typeof uuid !== 'undefined' && typeof num_frames !== 'undefined') {
    changeBackgroundImage(frame, uuid)
    changeClass(frame, num_frames)
  }
};

function setVideoInfo() {
  loadBackgroundImages(num_frames, uuid)
};

function uploadFile() {
  const form = document.getElementById('upload-form');
  const formData = new FormData(form);
  const fileLabel = document.querySelector('.file-label span');

  fileLabel.textContent = "Upload successful";
  window.scrollTo(0, document.documentElement.scrollHeight || document.body.scrollHeight);

  fetch('/upload', {
      method: 'POST',
      body: formData,
  })
  .then(response => {
      if (response.ok) {
          return response.json(); // Process JSON response
      }
      throw new Error('Network response was not ok.');
  })
  .then(data => {
      uuid = data.uuid;
      num_frames = data.num_frames;
      predicts = data.predicts;
      classes = data.classes;
      // setVideoInfo();
      var old_slider = document.querySelectorAll('.rangeslider');
      var old_number = document.querySelectorAll('.image-number');
      var old_classes = document.querySelectorAll('#predicts-list li');
      old_slider.forEach((slider_i) => {
        slider_i.classList.add("hidden");
      });
      old_number.forEach((number_i) => {
        number_i.classList.add("hidden");
      });
      old_classes.forEach((class_i) => {
        class_i.classList.add("hidden");
      });
      for (var i = 0; i < data.classes.length; i++) {
        $('#predicts-list').append('<li class="predict"><p class="emotion">' + data.classes[i] + '</p><p class="percent">' + 0 + '%</p></li>');
      };
      !$('#video-detailed').append('<input id="slider" type="range" min="1" max="' + num_frames + '" value="1" data-rangeslider>' +
        '<div class="image-number" id="image-number">1</div>');
      !function(a){"use strict";"function"==typeof define&&define.amd?define(["jquery"],a):"object"==typeof exports?module.exports=a(require("jquery")):a(jQuery)}(function(a){"use strict";function b(){var a=document.createElement("input");return a.setAttribute("type","range"),"text"!==a.type}function c(a,b){var c=Array.prototype.slice.call(arguments,2);return setTimeout(function(){return a.apply(null,c)},b)}function d(a,b){return b=b||100,function(){if(!a.debouncing){var c=Array.prototype.slice.apply(arguments);a.lastReturnVal=a.apply(window,c),a.debouncing=!0}return clearTimeout(a.debounceTimeout),a.debounceTimeout=setTimeout(function(){a.debouncing=!1},b),a.lastReturnVal}}function e(a){return a&&(0===a.offsetWidth||0===a.offsetHeight||a.open===!1)}function f(a){for(var b=[],c=a.parentNode;e(c);)b.push(c),c=c.parentNode;return b}function g(a,b){function c(a){"undefined"!=typeof a.open&&(a.open=a.open?!1:!0)}var d=f(a),e=d.length,g=[],h=a[b];if(e){for(var i=0;e>i;i++)g[i]=d[i].style.cssText,d[i].style.setProperty?d[i].style.setProperty("display","block","important"):d[i].style.cssText+=";display: block !important",d[i].style.height="0",d[i].style.overflow="hidden",d[i].style.visibility="hidden",c(d[i]);h=a[b];for(var j=0;e>j;j++)d[j].style.cssText=g[j],c(d[j])}return h}function h(a,b){var c=parseFloat(a);return Number.isNaN(c)?b:c}function i(a){return a.charAt(0).toUpperCase()+a.substr(1)}function j(b,e){if(this.$window=a(window),this.$document=a(document),this.$element=a(b),this.options=a.extend({},n,e),this.polyfill=this.options.polyfill,this.orientation=this.$element[0].getAttribute("data-orientation")||this.options.orientation,this.onInit=this.options.onInit,this.onSlide=this.options.onSlide,this.onSlideEnd=this.options.onSlideEnd,this.DIMENSION=o.orientation[this.orientation].dimension,this.DIRECTION=o.orientation[this.orientation].direction,this.DIRECTION_STYLE=o.orientation[this.orientation].directionStyle,this.COORDINATE=o.orientation[this.orientation].coordinate,this.polyfill&&m)return!1;this.identifier="js-"+k+"-"+l++,this.startEvent=this.options.startEvent.join("."+this.identifier+" ")+"."+this.identifier,this.moveEvent=this.options.moveEvent.join("."+this.identifier+" ")+"."+this.identifier,this.endEvent=this.options.endEvent.join("."+this.identifier+" ")+"."+this.identifier,this.toFixed=(this.step+"").replace(".","").length-1,this.$fill=a('<div class="'+this.options.fillClass+'" />'),this.$handle=a('<div class="'+this.options.handleClass+'" />'),this.$range=a('<div class="'+this.options.rangeClass+" "+this.options[this.orientation+"Class"]+'" id="'+this.identifier+'" />').insertAfter(this.$element).prepend(this.$fill,this.$handle),this.$element.css({position:"absolute",width:"1px",height:"1px",overflow:"hidden",opacity:"0"}),this.handleDown=a.proxy(this.handleDown,this),this.handleMove=a.proxy(this.handleMove,this),this.handleEnd=a.proxy(this.handleEnd,this),this.init();var f=this;this.$window.on("resize."+this.identifier,d(function(){c(function(){f.update(!1,!1)},300)},20)),this.$document.on(this.startEvent,"#"+this.identifier+":not(."+this.options.disabledClass+")",this.handleDown),this.$element.on("change."+this.identifier,function(a,b){if(!b||b.origin!==f.identifier){var c=a.target.value,d=f.getPositionFromValue(c);f.setPosition(d)}})}Number.isNaN=Number.isNaN||function(a){return"number"==typeof a&&a!==a};var k="rangeslider",l=0,m=b(),n={polyfill:!0,orientation:"horizontal",rangeClass:"rangeslider",disabledClass:"rangeslider--disabled",horizontalClass:"rangeslider--horizontal",verticalClass:"rangeslider--vertical",fillClass:"rangeslider__fill",handleClass:"rangeslider__handle",startEvent:["mousedown","touchstart","pointerdown"],moveEvent:["mousemove","touchmove","pointermove"],endEvent:["mouseup","touchend","pointerup"]},o={orientation:{horizontal:{dimension:"width",direction:"left",directionStyle:"left",coordinate:"x"},vertical:{dimension:"height",direction:"top",directionStyle:"bottom",coordinate:"y"}}};return j.prototype.init=function(){this.update(!0,!1),this.onInit&&"function"==typeof this.onInit&&this.onInit()},j.prototype.update=function(a,b){a=a||!1,a&&(this.min=h(this.$element[0].getAttribute("min"),0),this.max=h(this.$element[0].getAttribute("max"),100),this.value=h(this.$element[0].value,Math.round(this.min+(this.max-this.min)/2)),this.step=h(this.$element[0].getAttribute("step"),1)),this.handleDimension=g(this.$handle[0],"offset"+i(this.DIMENSION)),this.rangeDimension=g(this.$range[0],"offset"+i(this.DIMENSION)),this.maxHandlePos=this.rangeDimension-this.handleDimension,this.grabPos=this.handleDimension/2,this.position=this.getPositionFromValue(this.value),this.$element[0].disabled?this.$range.addClass(this.options.disabledClass):this.$range.removeClass(this.options.disabledClass),this.setPosition(this.position,b)},j.prototype.handleDown=function(a){if(this.$document.on(this.moveEvent,this.handleMove),this.$document.on(this.endEvent,this.handleEnd),!((" "+a.target.className+" ").replace(/[\n\t]/g," ").indexOf(this.options.handleClass)>-1)){var b=this.getRelativePosition(a),c=this.$range[0].getBoundingClientRect()[this.DIRECTION],d=this.getPositionFromNode(this.$handle[0])-c,e="vertical"===this.orientation?this.maxHandlePos-(b-this.grabPos):b-this.grabPos;this.setPosition(e),b>=d&&b<d+this.handleDimension&&(this.grabPos=b-d)}},j.prototype.handleMove=function(a){a.preventDefault();var b=this.getRelativePosition(a),c="vertical"===this.orientation?this.maxHandlePos-(b-this.grabPos):b-this.grabPos;this.setPosition(c)},j.prototype.handleEnd=function(a){a.preventDefault(),this.$document.off(this.moveEvent,this.handleMove),this.$document.off(this.endEvent,this.handleEnd),this.$element.trigger("change",{origin:this.identifier}),this.onSlideEnd&&"function"==typeof this.onSlideEnd&&this.onSlideEnd(this.position,this.value)},j.prototype.cap=function(a,b,c){return b>a?b:a>c?c:a},j.prototype.setPosition=function(a,b){var c,d;void 0===b&&(b=!0),c=this.getValueFromPosition(this.cap(a,0,this.maxHandlePos)),d=this.getPositionFromValue(c),this.$fill[0].style[this.DIMENSION]=d+this.grabPos+"px",this.$handle[0].style[this.DIRECTION_STYLE]=d+"px",this.setValue(c),this.position=d,this.value=c,b&&this.onSlide&&"function"==typeof this.onSlide&&this.onSlide(d,c)},j.prototype.getPositionFromNode=function(a){for(var b=0;null!==a;)b+=a.offsetLeft,a=a.offsetParent;return b},j.prototype.getRelativePosition=function(a){var b=i(this.COORDINATE),c=this.$range[0].getBoundingClientRect()[this.DIRECTION],d=0;return"undefined"!=typeof a["page"+b]?d=a["client"+b]:"undefined"!=typeof a.originalEvent["client"+b]?d=a.originalEvent["client"+b]:a.originalEvent.touches&&a.originalEvent.touches[0]&&"undefined"!=typeof a.originalEvent.touches[0]["client"+b]?d=a.originalEvent.touches[0]["client"+b]:a.currentPoint&&"undefined"!=typeof a.currentPoint[this.COORDINATE]&&(d=a.currentPoint[this.COORDINATE]),d-c},j.prototype.getPositionFromValue=function(a){var b,c;return b=(a-this.min)/(this.max-this.min),c=Number.isNaN(b)?0:b*this.maxHandlePos},j.prototype.getValueFromPosition=function(a){var b,c;return b=a/(this.maxHandlePos||1),c=this.step*Math.round(b*(this.max-this.min)/this.step)+this.min,Number(c.toFixed(this.toFixed))},j.prototype.setValue=function(a){(a!==this.value||""===this.$element[0].value)&&this.$element.val(a).trigger("input",{origin:this.identifier})},j.prototype.destroy=function(){this.$document.off("."+this.identifier),this.$window.off("."+this.identifier),this.$element.off("."+this.identifier).removeAttr("style").removeData("plugin_"+k),this.$range&&this.$range.length&&this.$range[0].parentNode.removeChild(this.$range[0])},a.fn[k]=function(b){var c=Array.prototype.slice.call(arguments,1);return this.each(function(){var d=a(this),e=d.data("plugin_"+k);e||d.data("plugin_"+k,e=new j(this,b)),"string"==typeof b&&e[b].apply(e,c)})},"rangeslider.js is available in jQuery context e.g $(selector).rangeslider(options);"});
      $(function(){
        $('input[type="range"]').rangeslider({
          polyfill:false,
          onInit:function(){
            $('.image-number').text($('input[type="range"]').val());
          },
          onSlide:function(position, value){
            $('.image-number').text(value);
            updateVideoInfo(value);
          },
          onSlideEnd:function(position, value){
          }
        });
      });
      $(document).ready(function() {
        $(".rangeslider").on({
          mouseenter: function(){
              $(".figure-class").show(); // Show element2 when hovering over element1
          },
          mouseleave: function(){
              $(".figure-class").hide(); // Hide element2 when mouse leaves element1
          }
      });
      });
      updateVideoInfo(1);
      if (data.success) {
        fileLabel.textContent = "Choose Video File";
      } else {
        fileLabel.textContent = "Upload failed: " + data.error;
      }
  })
  .catch(error => {
    fileLabel.textContent = "Error: " + error.message;
  });
}