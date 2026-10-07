(() => {
  const root=document.getElementById('leav-removals'),canvas=root.querySelector('.rm-canvas'),svg=canvas.querySelector('svg');
  const play=root.querySelector('[data-play]'),action=root.querySelector('[data-action]'),reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const NS='http://www.w3.org/2000/svg',nodes={},edges={},masks=[];
  const assets={audio:'assets/method/guitar-mel.png',video:'assets/method/guitar-frame-0.jpg',plane:'assets/attention/c4r8Kd6Q1n8_000008/poster.jpg',piano:'assets/attention/M4rXhyyvERM_000040/poster.jpg'};
  const clamp=x=>Math.max(0,Math.min(1,x)),ease=x=>{x=clamp(x);return x*x*(3-2*x)},mix=(a,b,t)=>a+(b-a)*t;
  const T={decoder:6000,negatives:9000,jepa:12000,jepaIn:1200,teacher:16500,predictor:19400,ours:22300,oursIn:1500,end:24200};
  let time=reduced.matches?T.end:0,playing=false,visible=false,started=false,raf=0,last=0,layout=null,lastWidth=0,lastStage=-1;
  function el(tag,attrs={},parent=svg) {const n=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);parent.append(n);return n;}
  function text(parent,value,x,y,cls='',size) {const n=el('text',{x,y,class:cls},parent);n.textContent=value;if(size)n.style.fontSize=size+'px';return n;}
  function rect(parent,x,y,w,h,cls='rm-box',rx=4) {return el('rect',{x,y,width:w,height:h,rx,class:cls},parent);}
  // All external connections are straight or use the same six-unit elbow.
  function route(points) {
    let d=`M${points[0].join(',')}`;
    for(let i=1;i<points.length-1;i++) {
      const a=points[i-1],b=points[i],c=points[i+1],ab=Math.hypot(b[0]-a[0],b[1]-a[1]),bc=Math.hypot(c[0]-b[0],c[1]-b[1]),r=Math.min(6,ab/2,bc/2);
      if(!ab||!bc)continue;
      const p=[b[0]+(a[0]-b[0])*r/ab,b[1]+(a[1]-b[1])*r/ab],q=[b[0]+(c[0]-b[0])*r/bc,b[1]+(c[1]-b[1])*r/bc];
      d+=` L${p.join(',')} Q${b.join(',')} ${q.join(',')}`;
    }
    return d+` L${points.at(-1).join(',')}`;
  }
  function line(parent,points,color='normal',arrow=true) {
    return el('path',{d:route(points),class:'rm-wire',...(color!=='normal'?{style:`stroke:${color==='blue'?'#2859a0':'#98685d'}`} : {}),...(arrow?{'marker-end':`url(#rm-${color==='normal'?'':color+'-'}arrow)`}:{})},parent);
  }
  function sample(parent,type,x,y,w,h,{masked=false,restored=false}={}) {
    const g=el('g',{'data-sample':type},parent);
    if(type==='joint') {
      sample(g,'video',x,y,w*.53,h);sample(g,'audio',x+w*.53+2,y,w*.47-2,h);return g;
    }
    el('image',{href:assets[type],x,y,width:w,height:h,preserveAspectRatio:type==='audio'?'none':'xMidYMid slice'},g);
    rect(g,x,y,w,h,'',2).setAttribute('style','fill:none;stroke:#202831;stroke-opacity:.12');
    if(masked||restored) {
      const patches=el('g',{},g),cols=type==='audio'?6:4,rows=type==='audio'?3:3;
      for(let r=0;r<rows;r++)for(let c=0;c<cols;c++) {
        if((c+2*r)%3===1||(c===2&&r===1)) {
          const p=rect(patches,x+c*w/cols+.5,y+r*h/rows+.5,w/cols-1,h/rows-1,'',0);
          p.setAttribute('style',masked?'fill:#e6e9ee;stroke:white;stroke-width:.6':'fill:#f0f5fc;fill-opacity:.05;stroke:#7d9dc7;stroke-width:1');
        }
      }
      if(masked)masks.push(patches);
    }
    return g;
  }
  function vector(parent,x,y,w,values,color='#2859a0') {
    const gap=4,bw=(w-gap*(values.length-1))/values.length;
    values.forEach((v,i)=>{el('rect',{x:x+i*(bw+gap),y:y+(24-v)/2,width:bw,height:v,rx:2,fill:color,opacity:.38+i*.08},parent);});
  }
  function node(id,x,y,w,h,kind,draw) {
    const g=el('g',{'data-node':id,class:'rm-node'},root.querySelector('[data-nodes]'));
    nodes[id]={g,x,y,w,h};g.setAttribute('transform',`translate(${x},${y})`);g.dataset.kind=kind;
    if(kind==='panel')rect(g,-w/2,-h/2,w,h,'rm-panel',6);
    draw(g,w,h);
    if(['decoder','negatives','teacher','predictor'].includes(id)) {
      el('circle',{cx:w/2-1,cy:-h/2+1,r:9,class:'rm-minus'},g);
      el('path',{d:`M${w/2-5},${-h/2+1}h8`,stroke:'white','stroke-width':1.5,class:'rm-minus',style:'fill:none'},g);
    }
  }
  function block(id,x,y,w,h,label,core=false) {
    node(id,x,y,w,h,'block',g=>{rect(g,-w/2,-h/2,w,h,core?'rm-core':'rm-box');label.split('|').forEach((v,i,ls)=>text(g,v,0,(i-(ls.length-1)/2)*19,core?'rm-core-label':'rm-title'));});
  }
  function input(id,x,y,w,h,label,masked=false) {
    node(id,x,y,w,h,'sample',g=>{sample(g,id==='target'?'joint':id,-w/2,-h/2,w,h,{masked});text(g,label,0,id==='target'?h/2+13:-h/2-13,'rm-caption');});
  }
  function reconstruction(x,y,w,h,mobile) {
    node('decoder',x,y,w,h,'panel',g=>{
      text(g,'Reconstruction',0,-h/2+18,'rm-title');
      if(mobile) {
        rect(g,-46,-h/2+43,92,44);text(g,'Decoder',0,-h/2+65);
        line(g,[[0,-h/2+91],[0,-h/2+116]]);
        sample(g,'audio',-46,-h/2+124,92,40,{restored:true});sample(g,'video',-29,-h/2+177,58,54,{restored:true});
        text(g,'Fill masked patches',0,h/2-14,'rm-caption',10);
      } else {
        rect(g,-w/2+20,-16,94,48);text(g,'Decoders',-w/2+67,8);
        line(g,[[-w/2+120,8],[-w/2+146,8]]);
        sample(g,'audio',-w/2+156,-22,154,52,{restored:true});sample(g,'video',w/2-112,-26,70,60,{restored:true});
        text(g,'Reconstruct missing patches',72,h/2-13,'rm-caption');
      }
    });
  }
  function contrastive(x,y,w,h,mobile) {
    node('negatives',x,y,w,h,'panel',g=>{
      text(g,mobile?'Contrastive':'Contrastive pairs',0,-h/2+18,'rm-title');
      if(mobile) {
        sample(g,'video',-28,-h/2+45,56,40);text(g,'Same clip',0,-h/2+36,'rm-caption',10);
        line(g,[[0,-h/2+90],[0,-h/2+99]],'blue');line(g,[[0,-h/2+117],[0,-h/2+108]],'blue');text(g,'Pull',26,-h/2+103,'rm-caption',10);
        sample(g,'audio',-36,-h/2+127,72,34);
        line(g,[[0,-h/2+180],[0,-h/2+169]],'red');line(g,[[0,-h/2+188],[0,-h/2+199]],'red');text(g,'Push',29,-h/2+184,'rm-caption',10);
        sample(g,'plane',-38,-h/2+206,36,28);sample(g,'piano',2,-h/2+206,36,28);
        text(g,'Other clips · negatives',0,h/2-14,'rm-caption',9.5);
      } else {
        sample(g,'video',-w/2+22,-23,72,52);text(g,'Same clip',-w/2+58,h/2-16,'rm-caption');
        sample(g,'audio',-w/2+174,-23,100,52);text(g,'Audio',-w/2+224,h/2-16,'rm-caption');
        sample(g,'plane',w/2-142,-23,58,52);sample(g,'piano',w/2-78,-23,58,52);text(g,'Other clips · negatives',w/2-81,h/2-16,'rm-caption');
        line(g,[[-w/2+103,5],[-w/2+124,5]],'blue');line(g,[[-w/2+163,5],[-w/2+142,5]],'blue');
        text(g,'Pull together',-w/2+133,-12,'rm-caption',10);
        line(g,[[-w/2+300,5],[-w/2+283,5]],'red');line(g,[[-w/2+311,5],[-w/2+328,5]],'red');
        text(g,'Push apart',-w/2+305,-12,'rm-caption',10);
      }
    });
  }
  function prediction(x,y,w,h) {
    node('latent',x,y,w,h,'panel',g=>{
      text(g,'Latent prediction',0,-h/2+20,'rm-title');
      vector(g,-42,-h/2+43,84,[12,22,16,8,19,14]);vector(g,-42,h/2-39,84,[14,20,18,10,19,12]);
      line(g,[[0,-7],[0,11]],'blue');text(g,'Predict',-66,-h/2+55,'rm-caption',10);text(g,'Target',-66,h/2-27,'rm-caption',10);
    });
  }
  function objective(x,y,w,h) {
    node('objective',x,y,w,h,'panel',g=>{
      g.firstChild.setAttribute('style','fill:var(--ours);stroke:#c7d6eb');
      text(g,'Invariance + SIGReg',0,-h/2+24,'rm-title').style.fill='var(--accent)';
      ['A','V','AV'].forEach((v,i)=>{text(g,v,-w/2+28,-25+i*35,'rm-caption',10);vector(g,-w/2+50,-37+i*35,w-94,[13,21,17,9,20,14]);});
    });
  }
  function build() {
    const mobile=canvas.clientWidth<680;
    layout={mobile,w:mobile?344:860,h:mobile?490:340};lastWidth=canvas.clientWidth;
    root.dataset.layout=mobile?'vertical':'horizontal';
    root.querySelector('[data-nodes]').replaceChildren();root.querySelector('[data-wires]').replaceChildren();
    for(const id of Object.keys(nodes))delete nodes[id];for(const id of Object.keys(edges))delete edges[id];masks.length=0;
    if(!mobile) {
      input('audio',64,103,88,52,'Audio',true);input('video',64,235,88,60,'Video',true);input('joint',64,247,88,52,'Audio + video');
      block('encA',207,103,128,52,'Audio encoder',true);block('encV',207,235,128,52,'Video encoder',true);
      block('shared',207,169,128,184,'Shared|encoder',true);
      reconstruction(580,101,512,134,false);contrastive(580,249,512,134,false);
      block('predictor',419,108,114,52,'Predictor');block('teacher',419,229,114,52,'EMA teacher');input('target',419,300,74,38,'Target view');
      prediction(646,169,206,190);block('projector',419,169,114,52,'Projector');objective(646,169,234,186);
    } else {
      input('audio',86,48,82,42,'Audio',true);input('video',258,48,82,42,'Video',true);input('joint',280,50,72,42,'Audio + video');
      block('encA',86,126,120,44,'Audio encoder',true);block('encV',258,126,120,44,'Video encoder',true);
      block('shared',172,150,248,56,'Shared encoder',true);
      reconstruction(88,335,148,260,true);contrastive(256,335,148,260,true);
      block('predictor',88,247,122,48,'Predictor');block('teacher',256,247,122,48,'EMA teacher');input('target',284,327,68,42,'Target view');
      prediction(120,390,206,136);block('projector',172,246,114,44,'Projector');objective(172,381,248,164);
    }
    const ema=el('text',{class:'rm-caption','data-ema-label':'',x:mobile?281:316,y:mobile?203:204},root.querySelector('[data-wires]'));ema.textContent='EMA';
    paint();
  }
  function show(id,alpha,x=nodes[id].x,y=nodes[id].y,cut=0,fade=0) {
    const n=nodes[id];n.g.style.display=alpha<.004?'none':'';n.g.setAttribute('opacity',alpha);n.g.dataset.removing=String(cut>0);
    n.g.setAttribute('transform',`translate(${x},${y-5*fade}) scale(${1-.04*fade})`);
  }
  function wire(id,points,alpha=1,{arrow=true,dashed=false}={}) {
    if(!edges[id])edges[id]=line(root.querySelector('[data-wires]'),points,'normal',arrow);
    const p=edges[id];p.setAttribute('d',route(points));p.setAttribute('opacity',alpha);p.style.display=alpha<.004?'none':'';if(dashed)p.dataset.ema='true';
  }
  function removal(start) {const strike=clamp((time-start)/600),fade=ease((time-start-1350)/700);return {strike,fade,a:1-fade};}
  function paint() {
    if(!layout)return;
    const j=ease((time-T.jepa)/T.jepaIn),o=ease((time-T.ours)/T.oursIn),d=removal(T.decoder),c=removal(T.negatives),t=removal(T.teacher),p=removal(T.predictor),m=layout.mobile;
    const stage=time<T.jepa?0:time<T.ours?1:2;
    const ax=m?mix(86,64,o):64,ay=m?48:mix(103,91,o),vx=m?mix(258,172,o):64,vy=m?48:mix(235,169,o);
    show('audio',1,ax,ay);show('video',1,vx,vy);show('joint',o);
    show('encA',1-j,m?mix(86,172,j):207,m?mix(126,150,j):mix(103,169,j));
    show('encV',1-j,m?mix(258,172,j):207,m?mix(126,150,j):mix(235,169,j));
    show('shared',j);show('decoder',d.a,undefined,undefined,d.strike,d.fade);show('negatives',c.a,undefined,undefined,c.strike,c.fade);
    show('predictor',j*p.a,undefined,undefined,p.strike,p.fade);show('teacher',j*t.a,undefined,undefined,t.strike,t.fade);show('target',j*t.a);show('latent',j*(1-o));show('projector',o);show('objective',o);
    masks.forEach(mask=>mask.setAttribute('opacity',1-j));
    if(!m) {
      wire('audioA',[[114,103],[137,103]],1-j);wire('videoV',[[114,235],[137,235]],1-j);
      wire('features',[[277,103],[290,103],[290,235],[277,235]],c.a,{arrow:false});
      wire('branch',[[290,169],[307,169]],c.a,{arrow:false});
      wire('decode',[[307,169],[307,101],[318,101]],d.a);wire('contrast',[[307,169],[307,249],[318,249]],c.a);
      wire('audioShared',[[114,ay],[137,ay]],j);wire('videoShared',[[114,vy],[137,vy]],j);wire('jointShared',[[114,247],[137,247]],o);
      wire('online',[[277,108],[356,108]],j*p.a);wire('predict',[[482,108],[537,108]],j*p.a*(1-o));
      wire('ema',[[277,229],[356,229]],j*t.a,{dashed:true});wire('target',[[419,275],[419,261]],j*t.a);
      wire('teacherLoss',[[482,229],[537,229]],j*t.a*(1-o));
      wire('project',[[277,169],[356,169]],o);wire('objective',[[482,169],[523,169]],o);
    } else {
      wire('audioA',[[86,75],[86,98]],1-j);wire('videoV',[[258,75],[258,98]],1-j);
      wire('features',[[86,154],[86,177],[258,177],[258,154]],c.a,{arrow:false});
      wire('decode',[[88,177],[88,199]],d.a);wire('contrast',[[256,177],[256,199]],c.a);
      wire('audioShared',[[ax,75],[ax,116]],j);
      wire('videoShared',[[vx,75],[vx,116]],j);
      wire('jointShared',[[280,77],[280,116]],o);
      wire('online',[[116,184],[116,200],[88,200],[88,217]],j*p.a);
      wire('ema',[[228,184],[228,200],[256,200],[256,217]],j*t.a,{dashed:true});
      wire('predict',[[88,277],[88,316]],j*p.a*(1-o));
      wire('teacherLoss',[[230,277],[230,302],[187,302],[187,316]],j*t.a*(1-o));
      wire('target',[[284,300],[284,277]],j*t.a);
      wire('project',[[172,184],[172,218]],o);wire('objective',[[172,274],[172,293]],o);
    }
    const ema=root.querySelector('[data-ema-label]');ema.setAttribute('opacity',j*t.a);
    canvas.style.aspectRatio=`${layout.w} / ${layout.h}`;svg.setAttribute('viewBox',`0 0 ${layout.w} ${layout.h}`);
    let message='';
    if(time>=T.decoder&&time<T.decoder+2600)message='Remove reconstruction decoders';
    if(time>=T.negatives&&time<T.negatives+2600)message='Remove contrastive negatives';
    if(time>=T.teacher&&time<T.teacher+2500)message='Remove the EMA teacher';
    if(time>=T.predictor&&time<T.predictor+2500)message='Remove the predictor';
    if(action.textContent!==message)action.textContent=message;
    if(lastStage!==stage) {
      root.dataset.stage=stage;lastStage=stage;
      root.querySelectorAll('[data-method]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.method===stage)));
      root.querySelector('[data-status]').textContent=['Reconstruction recovers masked samples. Contrastive learning pulls matching clips together and pushes other clips apart.','MJEPA: a shared encoder, an EMA teacher, and a predictor.','LeAVJEPA: a shared encoder and projector, with invariance plus SIGReg. Reconstruction, negatives, the teacher and predictor are removed.'][stage];
    }
    root.dataset.time=Math.round(time);
  }
  function controls(){play.textContent=playing?'Pause':time>=T.end?'Replay':'Play';}
  function tick(now){raf=0;if(!root.isConnected)return;const dt=last?Math.min(60,now-last):0;last=now;if(!playing||!visible||document.hidden)return;time=Math.min(T.end,time+dt);paint();if(time===T.end){playing=false;controls();return;}raf=requestAnimationFrame(tick);}
  function start(){if(!raf){last=0;raf=requestAnimationFrame(tick);}}
  function replay(){time=0;started=true;playing=!reduced.matches;paint();controls();if(playing)start();}
  play.addEventListener('click',()=>{if(time>=T.end)replay();else{playing=!playing;controls();if(playing)start();}});
  root.querySelectorAll('[data-method]').forEach(b=>b.addEventListener('click',()=>{playing=false;started=true;time=[0,T.jepa+T.jepaIn+200,T.end][+b.dataset.method];paint();controls();}));
  new ResizeObserver(()=>{if(canvas.clientWidth!==lastWidth)build();}).observe(canvas);build();controls();
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible&&entries[0].intersectionRatio>=.6&&!started&&!reduced.matches)replay();else if(visible&&playing)start();},{threshold:[0,.1,.6]}).observe(root);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden&&playing)start();});
  reduced.addEventListener('change',()=>{if(reduced.matches){playing=false;time=T.end;paint();controls();}});
})();
