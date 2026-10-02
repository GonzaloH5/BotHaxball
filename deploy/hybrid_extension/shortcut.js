// Shared shortcut contract for MAIN, settings and the worker (which labels UI messages).
(function(root,factory){const api=factory();if(typeof module==='object')module.exports=api;else root.RS4Shortcut=api;})(globalThis,()=>{
  'use strict';
  function normalize(value){
    if(typeof value==='string')value={code:value}; // existing installations
    if(!value||typeof value.code!=='string'||!/^\w{1,40}$/.test(value.code)
      ||/^(?:Escape|Unidentified|Dead|Process|Control(?:Left|Right)|Shift(?:Left|Right)|Alt(?:Left|Right)|Meta(?:Left|Right))$/.test(value.code))return null;
    return {code:value.code,ctrl:!!value.ctrl,alt:!!value.alt,shift:!!value.shift,meta:!!value.meta};
  }
  function fromEvent(event){
    if(event.repeat||event.isComposing)return null;
    return normalize({code:event.code,ctrl:event.ctrlKey,alt:event.altKey,shift:event.shiftKey,meta:event.metaKey});
  }
  function matches(event,value){
    const shortcut=normalize(value);
    return !!shortcut&&!event.isComposing&&event.code===shortcut.code
      &&!!event.ctrlKey===shortcut.ctrl&&!!event.altKey===shortcut.alt
      &&!!event.shiftKey===shortcut.shift&&!!event.metaKey===shortcut.meta;
  }
  function label(value){
    const shortcut=normalize(value);if(!shortcut)return 'Sin atajo';
    const key=shortcut.code.replace(/^Key/,'').replace(/^Digit/,'').replace(/^Numpad/,'Num ');
    return [...(shortcut.ctrl?['Ctrl']:[]),...(shortcut.alt?['Alt']:[]),...(shortcut.shift?['Shift']:[]),...(shortcut.meta?['Win/Meta']:[]),key].join(' + ');
  }
  return {normalize,fromEvent,matches,label};
});
