// ==UserScript==
// @name        New script - harvard.edu
// @namespace   Violentmonkey Scripts
// @match       http://ask-dl.fas.harvard.edu/content/*
// @grant       none
// @version     1.0
// @author      -
// @description 01/08/2020 15:19:26
// ==/UserScript==

/*
 for (i = 0; i < document.getElementsByTagName("a").length; i++) {

                console.log(document.getElementsByTagName("a")[i].href);

  }
*/


listaLinks = [

'10-eji-obgbe'
,'10-eji-ogbe-oyinbo'
,'10-eji-ogbe-2'
,'12-ogbe-y-ku'
,'13-ogbe-wo-hin-iwori'
,'14-ogbe-di'
,'15-ogbe-dawo-osun-irosun'
,'16-ogbe-hun-le-w-nrin'
,'17-ogbe-bara'
,'18-ogbe-kanran'
,'19-ogbe-yonu-ogunda'
,'110-ogbe-ri-ku-sa-sa'
,'111-ogbe-ka'
,'112-ogbe-t-m-p-n-oturup-n'
,'113-ogbe-alara-otura'
,'114-ogbe-wat-ir-t'
,'115-ogbe-s'
,'116-ogbe-fun'

,'20-oy-ku-meji-ii'
,'20-protective-medicine'
,'21-oy-ku-logbe-ogbe'
,'23-oy-ku-biwori-iwori'
,'24-oy-ku-di-odi'
,'25-oy-ku-rosun-irosun'
,'26-oy-w-nrin-w-nrin'
,'27-oy-ku-pabala-bara'
,'28-oy-ku-p-l-kan-kanran'
,'29-oy-ku-j-ko-da-ogunda'
,'210-oy-ku-teko-asa-sa'
,'211-oy-ku-bi-ka-ika'
,'212-oy-ku-ba-turup-n-oturup-n'
,'213-oy-ku-tia-otura'
,'214-p-ku-r-t-ir-t'
,'215-oy-ku-pa-akin-s-s'
,'216-ya-foku-ofun'

,'30-iwori-meji-ii'
,'31-iwori-bogbe-ogbe'
,'32-iwori-oy-ku'
,'34-iwori-di-odi'
,'35-iwori-rosun-irosun'
,'36-iwori-war-rin-w-nrin'
,'37-iwori-bara'
,'38-iwori-kanran'
,'39-iwori-ogun-tan-ogunda'
,'310-iwori-w-sa-sa'
,'311-iwori-ay-ka-ika'
,'312-iwori-tutu-oturup-n'
,'313-iwori-woti-otura'
,'314-iwori-wol-aat-ir-t'
,'315-iwori-w-s-s'
,'316-iwori-wofun-ofun'


,'40-odi-meji'
,'41-idin-gbe-ogbe'
,'42-idin-y-ku-oy-ku'
,'43-idin-wori-iwori'
,'45-idin-osun-irosun'
,'46-idin-aarin-w-nrin'
,'47-idin-bara-bara'
,'48-idin-kanran-kanran'
,'49-idin-gunda-ogunda'
,'410-idin-sa-sa'
,'411-idin-ka-ika'
,'412-idin-turu-oturup-n'
,'413-idin-atago-otura'
,'414-idin-l-k-ir-t'
,'415-idin-s-s'
,'416-idin-fun-fin-ofun'

,'50-irosun-meji'
,'51-irosun-agbe-ogbe'
,'52-irosun-oy-gun-oy-ku'
,'53-irosun-aw-nye-iwori'
,'54-irosun-odi'
,'56-irosun-l-rin-w-nrin'
,'57-irosun-bara'
,'58-irosun-kanran'
,'59-irosun-g-da-ogunda'
,'510-irosun-sa'
,'511-iro-mo-sun-ka-ika'
,'512-irosun-tutu-oturup-n'
,'513-irosun-la-otura'
,'514-irosun-opin-mi-ir-t-0'
,'515-irosun-oso-s'
,'516-irosun-afin-ofun'

,'60-w-nrin-meji'
,'61-w-nrin-sogbe-ogbe'
,'62-w-nrin-yapin-oy-ku'
,'63-w-nrin-were-iwori'
,'64-w-nrin-wodi-odi'
,'65-w-nrin-anrosun-irosun'
,'67-w-nrin-l-ba-bara'
,'68-w-nrin-k-ran-kanran'
,'69-w-nrin-dagbun-ogunda'
,'610-w-nrin-wosa-sa'
,'611-w-nrin-woka-ika'
,'612-w-nrin-baturup-n-oturup-n'
,'613-w-nrin-batura-otura'
,'614-w-nrin-yanr-t-ir-t'
,'615-w-nrin-was-s'
,'616-w-nrin-wofun-ofun'

,'70-bara-meji'
,'71-bara-gbogbe-ogbe'
,'72-bara-oy-ku'
,'73-bara-nikosi-iwori'
,'74-bara-bodi-odi'
,'75-bara-kosun-irosun'
,'76-bara-w-nrin'
,'78-bara-kanran'
,'79-bara-gun-tan-ogunda'
,'710-bara-sa'
,'711-bara-ka-ika'
,'712-bara-tutu-oturup-n'
,'713-bara-otura'
,'714-bara-r-t-ir-t'
,'715-bara-s'
,'716-bara-ofun'

,'80-kanran-meji'
,'81-kanran-sode-ogbe'
,'82-kanran-oy-ku'
,'83-kanran-iwori'
,'84-kanran-odi'
,'85-kanran-rosun-irosun'
,'86-kanran-w-nrin'
,'87-kanran-bara'
,'89-kanran-gun-tan-ogunda'
,'810-kanran-sa'
,'811-kanran-ka-ika'
,'812-kanran-tutu-oturup-n'
,'813-kanran-otura'
,'814-kanran-aat-ir-t'
,'815-kanran-ada-s-nu-s'
,'816-kanran-ofun'

,'90-ogunda-meji'
,'91-ogunda-mo-ro-gbe-ogbe'
,'92-ogunda-yapin-oy-ku'
,'93-odunga-wori-joko-iwori'
,'94-ogunda-di-odi'
,'95-ogunda-pa-ran-tan-o-f-j-kin-irosun-l-nu-irosun'
,'96-ogunda-w-n-l-j-aarin-w-nrin'
,'97-ogunda-ak-bara'
,'98-ogunda-kanran-kanran'
,'910-ogunda-m-sa-sa'
,'911-ogunda-wo-abe-aka-ika'
,'912-ogunda-ba-turup-n-oturup-n'
,'913-ogunda-t-tuala-otura'
,'914-ogunda-k-t-ir-t'
,'915-ogunda-s-s'
,'916-ogunda-fun-ofun'

,'100-sa-meji'
,'101-sa-lu-ogbe-ogbe'
,'102-sa-y-ku-oy-ku'
,'103-sa-wori-iwori'
,'104-sa-di-odi'
,'105-sa-le-sun-irosun'
,'106-sa-w-nrin-w-nrin'
,'107-sa-bara'
,'108-sa-kanran-kanran'
,'109-sa-gunda-ogunda'
,'1011-sa-ka-ika'
,'1012-sa-turup-n-oturup-n'
,'1013-sa-tua-otura'
,'1014-sa-bir-t-ir-t'
,'1015-sa-s-s'
,'1016-sa-fun-ofun'

,'110-ika-meji'
,'111-ika-gbe-mi-ogbe'
,'112-ika-y-ku-oy-ku'
,'113-ika-wori-iwori'
,'114-ika-di-odi'
,'115-ika-rosun-irosun'
,'116-ika-w-nrin-w-nrin'
,'117-ika-bara-bara'
,'118-ika-kanran-kanran'
,'119-ika-gunda-ogunda'
,'1110-ika-sa-sa'
,'1112-ika-turup-n-oturup-n'
,'1113-ika-tura-otura'
,'1114-ika-r-t-ir-t'
,'1115-ika-s-s'
,'1616-ika-fun-ofun'

,'120-l-gb-n-oturup-n-meji'
,'121-oturup-n-gbe-ogbe'
,'122-oturup-n-y-ku-oy-ku'
,'123-oturup-n-wori-iwori'
,'124-oturup-n-di-odi'
,'125-oturup-n-rosun-irosun-0'
,'126-oturup-n-w-nrin'
,'127-oturup-n-dara-bara'
,'128-otutu-ko-le-p-kanran-oturup-n-kanran'
,'129-oturup-n-guntan-ogunda'
,'1210-oturup-n-sa'
,'1211-oturu-p-n-ka-ika'
,'1213-oturup-n-otura'
,'1214-oturup-n-r-t-ir-t'
,'1215-oturup-n-s-s'
,'1216-oturup-n-fun-ofun'

,'130-otura-meji'
,'131-otura-ariko-ogbe'
,'132-otura-yapin-oy-ku'
,'133-otura-olonwo-iwori'
,'134-otura-di-odi'
,'135-otura-rosun-irosun'
,'136-otura-m-l-w-nrin'
,'137-otura-malabara-bara'
,'138-otura-tikun-kanran'
,'139-otura-rera-ogunda'
,'1310-otura-sa-sa'
,'1311-otura-tu-ka-ika'
,'1312-otura-ba-ti-oturup-n'
,'1314-otura-r-t-ir-t'
,'1315-otura-s-s'
,'1316-otura-fun-fun-ofun'

,'140-eji-l-m-r-ir-t-meji'
,'141-ir-t-agbe-ogbe'
,'142-aat-y-ku-ir-t-oy-ku'
,'143-ir-t-aw-nye-iwori'
,'144-ir-t-di-odi'
,'145-aat-rosun-ir-t-irosun'
,'146-ir-t-l-ta-w-nrin'
,'147-ir-t-bara'
,'148-ir-t-kanran'
,'149-ir-t-guntan-ogunda'
,'1410-ar-t-tan-nsa-ir-t-sa'
,'1411-aat-ka-ir-t-ika'
,'1412-ir-t-tutu-oturup-n'
,'1413-ir-t-otura'
,'1415-iru-kun-ir-t-s'
,'1416-irr-t-afin-ofun'

,'150-s-meji'
,'151-s-otolu-ogbe'
,'152-s-y-ku-oy-ku'
,'153-s-wori-iwori'
,'154-s-di-odi'
,'155-s-rosun-irosun'
,'156-s-oloogun-w-nrin'
,'157-s-bara-bara'
,'158-s-k-ran-kanran'
,'159-s-omolu-ogunda'
,'1510-s-saa-sa'
,'1511-s-ka-ika'
,'1512-s-turup-n-oturup-n'
,'1513-s-awurila-otura'
,'1514-s-bi-r-t-sile-aje'
,'1516-s-fun-ofun'

,'160-r-gun-ofun-meji'
,'161-ofun-agbe-ogbe'
,'162-ofun-t-na-oy-ku'
,'163-ofun-wori-iwori'
,'164-ofun-di-odi'
,'165-ofun-olosun-irosun'
,'166-ofun-w-nrin'
,'167-ofun-bara-bara'
,'168-ofun-kanran-kanran'
,'169-ofun-ko-ogunda'
,'1610-ofun-saa-sa'
,'1611-ofun-ka-ika'
,'1612-ofun-turup-n-oturup-n'
,'1613-ofun-tura-otura'
,'1614-ofun-r-t-ir-t'
,'1615-ofun-s-s'

    ]















function obterTranscricoes(){



var nomeArquivo = window.location.pathname.replace('/content/', '')
var proximaPagina = document.getElementsByClassName('page-next')[0]


//https://www.w3schools.com/js/js_array_iteration.asp
var i;
var interval = 5000;
var yorubaText = '';
var englishText = '';



  try{



var divEnglishParagrafos = document.getElementsByClassName("right odu-transcription rounded-small")[0].children
var divYorubaParagrafos = document.getElementsByClassName("left odu-transcription rounded-small")[0].children



     if(divYorubaParagrafos != null && divEnglishParagrafos != null){




             for (i = 0; i < divYorubaParagrafos.length; i++) {

                  yorubaText +=  divYorubaParagrafos[i].innerText.replace('<br>', '\n');

              }


            console.log(yorubaText);

            download( 'YORUBA_'+nomeArquivo+'_.txt', yorubaText);




                        for (i = 0; i < divEnglishParagrafos.length; i++) {

                              englishText +=  divEnglishParagrafos[i].innerText.replace('<br>', '\n');

                        }


           console.log(englishText);

           download('ENGLISH'+nomeArquivo+'_.txt', englishText);




    }


  }catch(e){


    console.log('Não possui texto em Yorubá')


  }finally{

  divYorubaParagrafos = null;
  divEnglishParagrafos = null;
  yorubaText = '';
  englishText = '';
  nomeArquivo = null;


  console.log('Proxima página: '+proximaPagina.href)

  setTimeout(function(){

         proximaPagina.click();

  }, interval)


  }

    }



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
        //pom.click();
    }
}





function imprimeLinks(){

    for (i = 24; i < 40; i++) {

                console.log(linksPagina[i].href);

  }



}







obterTranscricoes();

