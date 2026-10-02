(function(root,factory){const api=factory();if(typeof module==='object')module.exports=api;else root.RS4Control=api;})(globalThis,()=>{
  'use strict';
  class Arbiter{
    constructor({write,now=()=>performance.now(),notify=()=>{},backgroundBot=false}){
      this.write=write;this.now=now;this.notify=notify;this.mode='HUMANO';this.epoch=0;
      this.session='';this.ready=false;this.permission=false;this.focused=true;this.lastFrame=-1;
      this.lastApplied=-1;this.lastResponse=0;this.reason='';this.held=new Set();this.blocked=new Set();
      this.backgroundBot=backgroundBot;
    }
    transition(mode,reason=''){
      this.epoch++;this.mode=mode;this.reason=reason;this.lastApplied=-1;this.lastFrame=-1;
      this.lastResponse=this.now();this.blocked=new Set(this.held);
      try{this.write(0);}catch(e){this.mode='SUSPENDIDO';this.ready=false;this.reason=e.message;}
      this.notify(this.status());return this.mode;
    }
    status(){return {mode:this.mode,epoch:this.epoch,session:this.session,reason:this.reason};}
    prepareHuman(reason){
      if(this.mode!=='HUMANO')return this.transition('HUMANO',reason);
      // Preparation is not a control handoff: invalidate model responses without
      // clearing native manual movement or blocking keys the human is holding.
      this.epoch++;this.reason=reason;this.lastApplied=this.lastFrame=-1;this.lastResponse=this.now();
      this.notify(this.status());return this.mode;
    }
    setSession(session){if(session!==this.session){this.permission=false;this.ready=false;this.session=session;this.prepareHuman('Nueva sesión; confirmar permiso para el bot');}}
    toggle(){
      if(this.mode==='BOT')return this.transition('HUMANO');
      if(!this.ready||!this.permission||!this.focused){this.reason='Necesita modelo, permiso y foco';this.notify(this.status());return this.mode;}
      return this.transition('BOT');
    }
    key(code,down){
      if(down)this.held.add(code);else{this.held.delete(code);this.blocked.delete(code);}
      return this.mode==='HUMANO'&&!this.blocked.has(code);
    }
    suspend(reason){return this.transition('SUSPENDIDO',reason);}
    focus(active){
      this.focused=active;
      if(!active&&this.backgroundBot){
        if(this.mode==='BOT'){this.reason='Bot en segundo plano';this.notify(this.status());}
        else if(this.mode==='HUMANO')this.transition('HUMANO','Sin foco; teclas liberadas');
      }else if(!active)this.suspend('Pestaña sin foco');
      else if(this.mode==='SUSPENDIDO')this.transition('HUMANO');
      else if(this.mode==='BOT'){this.reason='';this.notify(this.status());}
    }
    accept(action){
      if(this.mode!=='BOT'||(!this.focused&&!this.backgroundBot)||action.session!==this.session||action.epoch!==this.epoch
        ||!Number.isInteger(action.frame)||action.frame<=this.lastApplied||!Number.isFinite(action.capturedAt)
        ||this.now()-action.capturedAt<0||this.now()-action.capturedAt>100
        ||!Number.isInteger(action.mask)||action.mask<0||action.mask>31)return false;
      try{this.write(action.mask);}catch(e){this.suspend(e.message);return false;}
      this.lastApplied=action.frame;this.lastResponse=this.now();return true;
    }
    watchdog(){if(this.mode==='BOT'&&this.now()-this.lastResponse>250)this.suspend('Inferencia sin respuesta válida');}
  }
  function captureControllers(scope,onCapture){
    const proto=scope.Function.prototype, originalBind=proto.bind;
    const target=scope.EventTarget.prototype, originalAdd=target.addEventListener;
    const bound=new WeakMap();let removed=false;
    function bind(...args){const result=Reflect.apply(originalBind,this,args);if(args[0]&&typeof args[0]==='object')bound.set(result,args[0]);return result;}
    function add(type,listener,...args){
      const result=Reflect.apply(originalAdd,this,[type,listener,...args]);
      if(this===scope.document&&type==='keydown'){
        const owner=bound.get(listener);if(owner)scope.queueMicrotask(()=>{if(!removed)onCapture(owner);});
      }
      return result;
    }
    proto.bind=bind;target.addEventListener=add;
    return ()=>{removed=true;if(proto.bind===bind)proto.bind=originalBind;if(target.addEventListener===add)target.addEventListener=originalAdd;};
  }
  return {Arbiter,captureControllers};
});
