// ==UserScript==
// @name        New script - bible.is
// @namespace   Violentmonkey Scripts
// @match       https://live.bible.is/bible/*
// @grant       none
// @version     1.0
// @author      -
// @description 04/08/2020 10:45:34
// ==/UserScript==


/*
//https://stackoverflow.com/questions/32601848/how-to-save-download-a-mp3-dataurl-object-using-java-script
window.onload = function() {
        document.getElementsByClassName('audio-player')[0].addEventListener('change', function(e){
            var file = e.target.files[0];
            console.log(e, file);
            if(file.type.match('audio.*')) { // if its an audio file
                document.getElementById( 'download' ).href = URL.createObjectURL(file);
            }
         });
    }

*/

/*
window.onload = function() {

var http = require('http');
var fs = require('fs');

var file = fs.createWriteStream(window.location.pathname + '.mp3');
var request = http.get(document.getElementsByClassName('audio-player')[0].src, function(response) {
  response.pipe(file);
});

}

*/


/*
//https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/downloads/download
function onStartedDownload(id) {
  console.log(`Started downloading: ${id}`);
}

function onFailed(error) {
  console.log(`Download failed: ${error}`);
}

var downloadUrl = document.getElementsByClassName('audio-player')[0].src;

var downloading = browser.downloads.download({
  url : downloadUrl,
  filename : window.location.pathname + '.mp3',
  conflictAction : 'uniquify'
});

downloading.then(onStartedDownload, onFailed);
*/

/*

//<a href="<url-goes-here>" data-downloadurl="audio/mpeg:<filename-goes-here>:<url-goes-here>" download="<filename-goes-here>">Click here to download the file</a>

function downloadArquivoAudio(){





var downloadUrlAudio = document.getElementsByClassName('audio-player')[0].src;
var nomeArquivoAudio = window.location.pathname + '.mp3';


//https://stackoverflow.com/questions/16267017/force-download-an-audio-stream
var createObjectURL = function (file) {
    if (window.webkitURL) {
        return window.webkitURL.createObjectURL(file);
    } else if (window.URL && window.URL.createObjectURL) {
        return window.URL.createObjectURL(file);
    } else {
        return null;
    }
},
xhr = new XMLHttpRequest();
xhr.open('GET', downloadUrlAudio, true);
xhr.responseType = 'blob';

xhr.onload = function (e) {
    if (this.status == 200) {
        var url = createObjectURL(new Blob([this.response], {
            type: 'audio/mpeg'
        }));
        var link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('Download', nomeArquivoAudio);
        link.appendChild(document.createTextNode('Download'));
        document.getElementsByTagName('body')[0].appendChild(link);
        link.click();

    }
};
xhr.send();


}

*/



/*

//https://stackoverflow.com/questions/3916191/download-data-url-file
function downloadAudiofile(url, filename) {

fetch(url).then(function(t) {
    return t.blob().then((b)=>{
        var a = document.createElement("a");
        a.href = URL.createObjectURL(b);
        a.setAttribute("download", filename);
        a.setAttribute("data-downloadurl", "audio/mpeg:"+url+":"+filename )
        a.click();
    }
    );
});
}

window.onload = function() {
  downloadAudiofile(downloadUrlAudio, nomeArquivoAudio);
}

*/



/*
//https://stackoverflow.com/questions/8236735/what-is-the-equivalent-of-wget-in-javascript-to-download-a-file-from-a-given-url
function downloadAudiofile() {
   window.open(downloadUrlAudio, 'Download');
 }

downloadAudiofile();
*/



//https://stackoverflow.com/questions/2897619/using-html5-javascript-to-generate-and-save-a-file
function download(filename, text) {
    var pom = document.createElement('a');
    pom.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    pom.setAttribute('download', filename);

    if (document.createEvent) {
        var event = document.createEvent('MouseEvents');
        event.initEvent('click', true, true);
        pom.dispatchEvent(event);
    }
    else {
        pom.click();
    }
}


function extrairTexto(){

  textoYoruba = document.getElementsByClassName('chapter')[0].textContent.replace('<br>', '\n').replace(new RegExp("[0-9]"), " ");

  console.log(textoYoruba);

  var nomeArquivoTexto = window.location.pathname + '.txt';

  download(nomeArquivoTexto, textoYoruba);


}


function extrairLinkAudio(){

  linkAudio = document.getElementsByTagName('audio')[0].src

  var nomeArquivoLinkAudio = window.location.pathname + '_linkAudio_.txt';

  download(nomeArquivoLinkAudio, linkAudio )


}


function clicarProximoLink(){

      //links = document.getElementsByClassName('chapter-box');
      //links[1].click();

    next = document.getElementsByClassName('chapter')[0].dataset.nextid;
  //https://stackoverflow.com/questions/19569656/javascript-regex-optional-character/19569846


  //1CO10
  regexp1 = /(\d+)(\w+)(\d+\d+)/
  resultado1 = next.match(regexp1);


  //1CO1
  regexp2 = /(\d+)(\w+)(\d+)/
  resultado2 = next.match(regexp2);


  //MAT1
  regexp3 = /(\w+)(\d+)/
  resultado3 = next.match(regexp3);

  //MAT10
  regexp4 = /(\w+)(\d+\d+)/
  resultado4 = next.match(regexp4);


  proximoLink = '';


    if(resultado1 ){
  //EXEMPLO: 1CO11
    proximoLink = next.replace(regexp1, "$1$2/$3")
  }


    if(resultado2 ){
    //EXEMPLO: 1CO1
    proximoLink = next.replace(regexp2, "$1$2/$3")
  }



  if(resultado3){
    //MAT1
    proximoLink = next.replace(regexp3, "$1/$2")
  }


    if(resultado4 ){
      //exemplo: MAT10
    proximoLink = next.replace(regexp4, "$1/$2")
  }


  console.log('R1: '+resultado1)
  console.log('R2: '+resultado2)
  console.log('R3: '+resultado3)
  console.log('R4: '+resultado4)
  console.log(proximoLink)





    location.assign('https://live.bible.is/bible/'+linguaAlvo+'/'+proximoLink);



  }

//linguaAlvo = 'PORB09'
linguaAlvo = 'YORYOR'

window.onload = function(){

  extrairTexto();
  extrairLinkAudio();


  setTimeout(clicarProximoLink(), 5000);


}

//window.close();



