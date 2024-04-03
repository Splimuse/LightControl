const WebS = require('ws');
const mqtt = require('mqtt');
// const express = require("express");

const webSocketServer = new WebS.Server({ port: 3000 });
// const app = express();
// app.use(express.static("public"));

const MQTT_IP_ADDR = process.env.MQTT_IP_ADDR || "192.168.0.105";
const MQTT_PORT = process.env.MQTT_PORT || 1883;


let arr = [
    '/devices/wb-mdm3_140/controls/K1',
    '/devices/wb-mdm3_140/controls/K2',
    '/devices/wb-mdm3_140/controls/K3',
    '/devices/wb-mr6c_237/controls/K1',
    '/devices/wb-mr6c_237/controls/K2',
    '/devices/wb-mr6c_237/controls/K3',
    '/devices/wb-mr6c_237/controls/K4',
    '/devices/wb-mr6c_237/controls/K5',
    '/devices/wb-mr6c_237/controls/K6',
    '/devices/wb-mdm3_140/controls/Channel 1',
    '/devices/wb-mdm3_140/controls/Channel 2',
    '/devices/wb-mdm3_140/controls/Channel 3',
]

let state = new Map([
    ['/devices/wb-mdm3_140/controls/K1', 0],
    ['/devices/wb-mdm3_140/controls/K2', 0],
    ['/devices/wb-mdm3_140/controls/K3', 0],
    ['/devices/wb-mr6c_237/controls/K1', 0],
    ['/devices/wb-mr6c_237/controls/K2', 0],
    ['/devices/wb-mr6c_237/controls/K3', 0],
    ['/devices/wb-mr6c_237/controls/K4', 0],
    ['/devices/wb-mr6c_237/controls/K5', 0],
    ['/devices/wb-mr6c_237/controls/K6', 0],
    ['/devices/wb-mdm3_140/controls/Channel 1', 1],
    ['/devices/wb-mdm3_140/controls/Channel 2', 1],
    ['/devices/wb-mdm3_140/controls/Channel 3', 1],
])

const mqttConnStr = `mqtt://${MQTT_IP_ADDR}:${MQTT_PORT}`;

console.log(`Connecting to mqtt ${mqttConnStr}`);
const mqttClient = mqtt.connect(mqttConnStr);

mqttClient.on('connect', function () {
    console.log("mqtt connection established")
    mqttClient.subscribe(arr);
});

let arrConn = []

mqttClient.on('message', (topic, message) => {
    let event = topic + ' - ' + message.toString();
    console.log(`mqtt event ${event}`);
    state.set(topic, message)
    arrConn.forEach(conn => {
        conn.send(event)
    })
});

webSocketServer.on('connection', webSocketConnection => {
    console.log(`client ${webSocketConnection} connected`)
    arrConn.push(webSocketConnection)
    state.forEach((value, key) => {
        let stateEvent = key + ' - ' + value.toString();
        console.log(`state event: ${stateEvent}`)
        webSocketConnection.send(stateEvent)
    })

    webSocketConnection.on('message', message => {
        let value = (Buffer.from(message)).toString();
        let arr = value.split(',');
        arr[1] = JSON.parse(arr[1]);
        if (arr[0].indexOf("Channel") != -1) {
            arr[1] = Number(arr[1])
        } else {
            arr[1] = Number(!arr[1]);
        }

        console.log("bright : " + arr[1]);
        mqttWrite(arr[0], arr[1])
    });
})

function mqttWrite(topic, value) {
    // if (topic.indexOf("Channel") != -1) {
    //     console.log(topic + ' !!!channel!!! ' + value);
    //     topic = topic + '/on';
    // } else {
    //     console.log(topic + ' !!!rele!!! ' + value);
    //     topic = topic + '/on';
    // }
    mqttClient.publish(topic + '/on', String(value));

}

// app.listen(8080);