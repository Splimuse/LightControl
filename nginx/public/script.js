const ws = new WebSocket('ws://localhost:8000/ws');
let a = document.querySelectorAll("a");
let title = document.querySelector("title");
const colorsPages = new Map([
    ["Главная", "#A28089"],
    ["Зона1", "#8458B3"],
    ["Зона2", "#F75990"],
    ["Зона3", "#FF9999"],
    ["Зона4", "#99FF66"],
]);



a.forEach(function (elem) {
    if (elem.innerText == title.innerText) {
        elem.style.color = colorsPages.get(title.innerText);
        elem.style.fontSize = "35px";
    }
})

ws.onopen = () => console.log('ONLINE');

ws.onclose = () => console.log('DISCONNECTED');

ws.onmessage = (msg) => {
    let res = msg.data.split(' - ');
    if (res[0].indexOf("Channel") != -1) {
        let name = "[name='" + res[0] + "']";
        try {
            let text = document.querySelector(name);
            text.innerText = res[1];
        } catch { console.log('error name'); }
        try {
            let channel = document.getElementById(res[0]);
            channel.value = res[1];
        } catch { console.log('error channel'); }


    } else {
        try {
            elem = document.getElementById(res[0]);
            elem.value = res[1];
        } catch { console.log('element not found') }
        let sorce = "[name='" + res[0] + "']";
        let bright = document.querySelector(sorce);
        let but = document.getElementById(res[0]);
        if (Number(res[1]) == 0) {
            try {
                bright.style.display = 'none';
            } catch {
                console.log("error");
            }
            try {
                but.style.background = "linear-gradient(white, rgb(175, 175, 163), silver)";
                but.style.boxShadow = "0 4px 12px hsla(0, 0%, 20%, .8)";
            } catch { }
        } else {
            try {
                bright.style.display = 'initial';
            } catch {
                console.log("error");
            }
            try {
                but.style.background = "linear-gradient(rgb(135, 135, 35), rgb(223, 236, 40), rgb(204, 194, 19))";
                but.style.boxShadow = "0px 4px 12px hsla(24, 97%, 49%, 0.973)";
            } catch { }
        }
    }


    console.log(res[0] + " ??? " + res[1]);


    console.log(res);

};

let buttons = document.querySelectorAll(".element_button > button");
console.log(`found buttons: ${buttons.length}`)
buttons.forEach(btn => {
    btn.addEventListener("click", () => {

        ws.send([btn.id, btn.value]);

    })
    console.log(`register click listener for ${btn}`)
})

let range = document.querySelectorAll(".range");
range.forEach(rng => {
    rng.onchange = () => {
        let topic = rng.id
        let rngValue = rng.value;
        ws.send([topic, rngValue]);
        console.log("range id = " + topic + " range value= " + rngValue)
    }

})




