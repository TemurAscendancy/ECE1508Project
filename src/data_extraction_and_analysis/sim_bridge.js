#!/usr/bin/env node
'use strict';
/**
 * Showdown needs .js to talk to it, sad...
 * perisistent bridge to the pokemon showdown simulator
 * Follows this structure:
 * in: {"id": "<battle id>", "op": "new" | "write" | "close", "data": "<sim input>"}
 * out: {"id": "<battle id>", "events": [...]}   or   {"id": ..., "error": "<msg>"}
 * ["update", <p1 view>, <p2 view>]   battle log, |split| lines resolved per player
 * ["side", "p1"|"p2", <text>]        |request|... or |error|... for that side
 * ["end", <json>]                    battle finished
 * ["log", <input log>]               answer to ">requestlog"
 */

const path = require('path');
const readline = require('readline');

// Anything the sim prints must not corrupt the stdout protocol.
console.log = (...args) => console.error(...args);

const root = path.resolve(process.argv[2] || '.');
const { BattleStream } = require(path.join(root, 'dist', 'sim', 'battle-stream.js'));
const { extractChannelMessages } = require(path.join(root, 'dist', 'sim', 'battle.js'));

const battles = new Map();

function makeStream() {
    const stream = new BattleStream({ keepAlive: true });
    stream.outbox = [];
    // Capture output synchronously instead of going through the async read side.
    stream.push = chunk => { stream.outbox.push(chunk); };
    stream.pushEnd = () => {};
    stream.pushError = err => { stream.outbox.push(`error\n${(err && err.stack) || err}`); };
    return stream;
}

function splitFirst(str, delim) {
    const i = str.indexOf(delim);
    if (i < 0) 
    {
        return [str, ''];
    }
    return [str.slice(0, i), str.slice(i + delim.length)];
}

function toEvents(outbox) {
    const events = [];
    for (const chunk of outbox) 
    {
        const [type, data] = splitFirst(chunk, '\n');
        if (type === 'update') 
        {
            const ch = extractChannelMessages(data, [1, 2]);
            events.push(['update', ch[1].join('\n'), ch[2].join('\n')]);
        } else if (type === 'sideupdate') 
        {
            const [side, text] = splitFirst(data, '\n');
            events.push(['side', side, text]);
        } else if (type === 'end') 
        {
            events.push(['end', data]);
        } else if (type === 'requesteddata') 
        {
            events.push(['log', data]);
        } else 
        {
            events.push([type, data]);
        }
    }
    return events;
}

function reply(obj) {
    process.stdout.write(JSON.stringify(obj) + '\n');
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on('line', line => {
    if (!line.trim()) return;
    let msg;
    try {
        msg = JSON.parse(line);
    } catch (e) {
        reply({ id: null, error: `bad json: ${e}` });
        return;
    }
    const { id, op, data } = msg;
    try {
        if (op === 'new') 
        {
            battles.set(id, makeStream());
            reply({ id, events: [] });
        } else if (op === 'close') 
        {
            battles.delete(id);
            reply({ id, events: [] });
        } else if (op === 'write') 
        {
            const stream = battles.get(id);
            if (!stream) throw new Error(`unknown battle id ${id}`);
            stream.outbox = [];
            stream._write(data);
            reply({ id, events: toEvents(stream.outbox) });
        } else 
        {
            throw new Error(`unknown op ${op}`);
        }
    } catch (e) 
    {
        reply({ id, error: String((e && e.stack) || e) });
    }
});
rl.on('close', () => process.exit(0));
