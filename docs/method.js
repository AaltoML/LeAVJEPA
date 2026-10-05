"use strict";
(() => {
  const root = document.getElementById("method-tour");
  if (!root) return;
  const asset = "assets/method/", A = "#e11d48", V = "#3b82f6", R = "#059669", ink = "#18181b";
  const duration = 18000;
  const stages = [
    [0, "Source"],
    [2500, "Patches"],
    [4000, "Views"],
    [8000, "Encoder"],
    [10500, "Alignment"],
    [13000, "SIGReg"],
    [16000, "Update"],
  ];
  const stops = [0, 3500, 7000, 9600, 12300, 15000, duration];
  const text = (x, y, value, attrs = "") => `<text x="${x}" y="${y}" ${attrs}>${value}</text>`;
  const rect = (x, y, w, h, fill, attrs = "") => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="${fill}" ${attrs}/>`;
  const dot = (x, y, r, fill, attrs = "") => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" ${attrs}/>`;
  const path = (d, attrs = "") => `<path d="${d}" fill="none" ${attrs}/>`;
  const image = (name, x, y, w, h, attrs = "") => `<image href="${asset}${name}" x="${x}" y="${y}" width="${w}" height="${h}" preserveAspectRatio="${name.includes("frame") ? "xMidYMid slice" : "none"}" ${attrs}/>`;
  const rowY = i => i * 73, viewW = 196;
  const views = [
    { name: "Global 1 · A + V", video: true, audio: true },
    { name: "Global 2 · A + V", video: true, audio: true },
    { name: "Local · audio", video: false, audio: true },
    { name: "Local · video", video: true, audio: false },
  ];
  const tokenStrip = (type, x) => Array.from({ length: 5 }, (_, k) => rect(x+k*12, 31, 8, 19, type === "video" ? V : A, 'class="ma-token"')).join("");
  const patches = (source, columns, rows, width, height) => {
    const w = width / columns, h = height / rows;
    return Array.from({length: columns * rows}, (_, i) => {
      const x = (i % columns) * w, y = Math.floor(i / columns) * h;
      return `<g transform="translate(${x+w/2} ${y+h/2})"><g class="ma-patch"><svg x="${-w/2}" y="${-h/2}" width="${w}" height="${h}" style="width:${w}px;height:${h}px" viewBox="${x} ${y} ${w} ${h}" overflow="hidden"><use href="#${source}"/></svg></g></g>`;
    }).join("");
  };
  const tubeletCells = [[132,12],[144,72],[72,108]];
  const batch = Array.from({ length: 40 }, (_, i) => {
    const r = Math.sqrt(-2*Math.log((i+.5)/40)), angle = i*2.39996323;
    return [.72*r*Math.cos(angle)+.45, 1.2*r*Math.sin(angle)-.25, r*Math.cos(angle), r*Math.sin(angle)];
  });
  root.innerHTML = `
    <svg class="ma-canvas" viewBox="0 0 1220 570" role="img" aria-labelledby="ma-title ma-desc">
      <title id="ma-title">The complete LeAVJEPA method, built from one guitar sample</title>
      <desc id="ma-desc">Frames and the spectrogram separate into patches. Two global views retain both modalities; two locals retain only audio or video. Each sequence starts with a labeled CLS token. A shared transformer repeats attention and MLP blocks, then its CLS output passes through the projector. Three stacked frames reveal temporal depth, with matching patches extending backward. Tubelets group patches from neighboring frame pairs. In the latent space, alignment brings views of one clip together. SIGReg compares characteristic functions of projected batches with the Gaussian target. The weighted losses update the shared encoder and projector.</desc>
      <defs>
        ${Array.from({length:3},(_,i)=>image(`guitar-frame-${i}.jpg`,0,0,168,168,`id="ma-frame-source-${i}"`)).join("")}
        ${image("guitar-mel.png",0,0,180,90,'id="ma-mel-source"')}
        <pattern id="ma-video-pattern" width="12" height="12" patternUnits="userSpaceOnUse">${path("M12 0H0V12", 'stroke="#ffffffc0" stroke-width="1.7"')}</pattern>
        <pattern id="ma-audio-pattern" width="3.6" height="11.25" patternUnits="userSpaceOnUse">${path("M3.6 0H0V11.25", 'stroke="#ffffff90" stroke-width=".5"')}</pattern>
        <clipPath id="ma-mel-clip"><rect id="ma-mel-wipe" width="0" height="90"/></clipPath>
        <marker id="ma-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">${path("M1 1L7 4L1 7", 'stroke="#a1a1aa" stroke-width="1.2"')}</marker>
      </defs>
      <g id="ma-connections" stroke="#a1a1aa" stroke-width="1.3" marker-end="url(#ma-arrow)"></g>
      <g id="ma-video">
        ${text(96,-12,"Video",'id="ma-video-label" text-anchor="middle" class="ma-kicker"')}
        ${[2,1,0].map(i=>`<g id="ma-frame-${i}" transform="translate(${i*12} ${28-i*14})" opacity="${1-i*.23}">
          ${rect(0,0,168,168,"white")}
          <g class="ma-frame-sheet"><use href="#ma-frame-source-${i}"/></g>
          <g class="ma-frame-patches">${patches(`ma-frame-source-${i}`,14,14,168,168)}</g>
          <g class="ma-frame-grid">${rect(0,0,168,168,"url(#ma-video-pattern)")}</g>
        </g>`).join("")}
        <g id="ma-tubelets">
          ${tubeletCells.map(([x,y])=>{
            const left=x+1.32,top=y+29.32,size=9.36;
            return `<g>
              ${path(`M${left} ${top}l12 -14h${size}v${size}l-12 14Z`, 'fill="#3b82f620" stroke="'+V+'" stroke-width="1.3"')}
              ${path(`M${left} ${top}h${size}v${size}h-${size}Z M${left+size} ${top}l12 -14`, 'stroke="'+V+'" stroke-width="1.5"')}
              ${path(`M${left+12} ${top-14}l12 -14h${size}v${size}l-12 14 M${left+12+size} ${top-14}l12 -14`, 'stroke="'+V+'" stroke-width="1" stroke-dasharray="2 2"')}
            </g>`;
          }).join("")}
          ${text(90,224,"Tubelets",'text-anchor="middle" class="ma-prominent"')}
        </g>
      </g>
      <g id="ma-audio">
        ${text(90,-12,"Waveform",'id="ma-audio-label" text-anchor="middle" class="ma-kicker"')}
        ${rect(0,0,180,90,"#fff1f2")}
        <g id="ma-wave">${image("guitar-wave.svg",0,12,180,66)}</g>
        <g id="ma-audio-sheet" clip-path="url(#ma-mel-clip)">${image("guitar-mel.png",0,0,180,90)}</g>
        <g id="ma-audio-patches">${patches("ma-mel-source",50,8,180,90)}</g>
        <g id="ma-audio-grid">${rect(0,0,180,90,"url(#ma-audio-pattern)")}</g>
        <g id="ma-scan">${path("M0 0V90",'stroke="#fb7185" stroke-width="2"')}</g>


      </g>
      <g id="ma-views">
        ${text(viewW/2,-26,"Views → tokens",'text-anchor="middle" class="ma-kicker"')}
        ${views.map((view,i)=>`<g id="ma-view-${i}" transform="translate(0 ${rowY(i)})">
          ${rect(0,0,viewW,62,"#ffffff",'stroke="#e4e4e7"')}${text(12,18,view.name,'class="ma-view-name"')}
          ${rect(10,29,46,23,"#eff6ff",'stroke="#bfdbfe"')}${text(33,45,"[CLS]",'text-anchor="middle" class="ma-cls-label"')}
          <g id="ma-row-${i}-video" class="ma-video-tokens">${tokenStrip("video",view.audio?68:128)}</g>
          <g id="ma-row-${i}-audio" class="ma-audio-tokens">${tokenStrip("audio",view.audio?128:68)}</g>
          ${!view.video||!view.audio ? `<g id="ma-drop-${i}">${rect(68,31,57,19,"#fafafa",'stroke="#d4d4d8" stroke-dasharray="3 3"')}${path("M91 36L101 45M101 36L91 45",'stroke="#d4d4d8"')}</g>` : ""}
        </g>`).join("")}


      </g>
      <g id="ma-model">
        ${text(85,-29,"Vision Transformer",'id="ma-model-title" text-anchor="middle" class="ma-kicker"')}
        ${rect(8,-8,170,100,"#fafafa",'stroke="#e4e4e7"')}${rect(4,-4,170,100,"#fafafa",'stroke="#d4d4d8"')}
        ${rect(0,0,170,100,"#fafafa",'stroke="#a1a1aa"')}
        <g id="ma-model-pulse">${rect(1,1,168,98,"#1d4ed80c",'stroke="#3b82f6"')}</g>
        ${rect(14,10,142,33,"white",'stroke="#d4d4d8"')}${text(85,32,"Attention",'text-anchor="middle" class="ma-block-label"')}
        ${path("M85 43V55",'stroke="#a1a1aa" marker-end="url(#ma-arrow)"')}
        ${rect(14,57,142,33,"white",'stroke="#d4d4d8"')}${text(85,79,"MLP",'text-anchor="middle" class="ma-block-label"')}
        <g id="ma-readout">
          ${path("M85 100V121M85 147V175",'stroke="#a1a1aa" marker-end="url(#ma-arrow)"')}
          ${rect(60,122,50,24,"#eff6ff",'stroke="#bfdbfe"')}${text(85,139,"[CLS]",'text-anchor="middle" class="ma-cls-label"')}
        </g>
        ${rect(0,176,170,46,"white",'stroke="#a1a1aa"')}<g id="ma-projector-pulse">${rect(1,177,168,44,"#eff6ff",'stroke="#93c5fd"')}</g>${text(85,204,"Projector",'text-anchor="middle"')}

      </g>
      <g id="ma-embeddings">
        ${text(105,-31,"Embeddings",'text-anchor="middle" class="ma-kicker"')}
        ${rect(0,0,210,224,"#ffffff",'stroke="#e4e4e7"')}
        <g transform="translate(105 103) scale(.84) translate(-125 -128)">
        ${path("M20 15V215H232",'stroke="#f4f4f5" stroke-width="1"')}
        <g id="ma-sigreg">
          <g id="ma-batch-drops" stroke="${R}" stroke-width=".65" stroke-opacity=".16"></g>
          <g id="ma-batch-original"></g><g id="ma-batch-axis" stroke="${R}" stroke-width="1.6"></g><g id="ma-batch-projected"></g>
        </g>
        <g id="ma-alignment">
          <g id="ma-align-lines" stroke-width="1.2" stroke-dasharray="3 4"></g>
          ${path("M125 120L133 128L125 136L117 128Z",'fill="white" stroke="'+ink+'" stroke-width="1.7"')}
        </g>
        <g id="ma-align-points"></g>
        </g>
        ${text(105,206,"Align the views",'id="ma-align-label" text-anchor="middle" class="ma-prominent"')}
      </g>
      <g id="ma-distribution">
        ${text(90,-31,"SIGReg",'text-anchor="middle" class="ma-kicker"')}
        ${text(90,-5,"Random projections",'text-anchor="middle" class="ma-prominent"')}
        ${path("M0 30H180",'stroke="#d1fae5"')}
        <g id="ma-projection-samples"></g>
        ${path("M90 42V63",'stroke="#a1a1aa" marker-end="url(#ma-arrow)"')}
        <g id="ma-histogram"></g>
        ${path("M0 176H180",'stroke="#d4d4d8"')}
        <path id="ma-gaussian" fill="none" stroke="${R}" stroke-width="2.3" pathLength="1"/>
        ${text(90,216,"Gaussian target",'text-anchor="middle" class="ma-prominent"')}
      </g>
    </svg>
    <div class="ma-timeline">
      <input id="ma-scrubber" type="range" min="0" max="${duration}" step="any" value="0" aria-label="Method animation timeline">
      <div class="ma-marks" role="group" aria-label="Jump to a point in the animation">
        ${stages.map(([, name], i) => `<button type="button" data-time="${stops[i]}" data-phase="${i}" style="--position:${stops[i]/duration*100}%" aria-label="Go to ${name.toLowerCase()}, ${stops[i]/1000} seconds"><i aria-hidden="true"></i><span>${name === "Alignment" ? "Align" : name === "Encoder" ? "Model" : name}</span></button>`).join("")}
      </div>
    </div>
`;

  const staticNodes = new Map([...root.querySelectorAll("[id]")].map(node => [node.id, node]));
  const get = id => staticNodes.get(`ma-${id}`) || root.querySelector(`#ma-${id}`);
  const videoFrames=Array.from({length:3},(_,i)=>({
    sheet:get(`frame-${i}`).querySelector(".ma-frame-sheet"),
    tiles:get(`frame-${i}`).querySelector(".ma-frame-patches"),
    grid:get(`frame-${i}`).querySelector(".ma-frame-grid"),
  }));
  const tubeletVolumes=[...get("tubelets").querySelectorAll("g")].map((node,i)=>({
    x:tubeletCells[i][0]+1.32,y:tubeletCells[i][1]+29.32,paths:[...node.querySelectorAll("path")],
  }));
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const mobile = matchMedia("(max-width: 620px)");
  const smooth = (t,a,b) => { const x=Math.max(0,Math.min(1,(t-a)/(b-a))); return x*x*(3-2*x); };
  const opacity = (id,value) => get(id).setAttribute("opacity",value.toFixed(3));
  let elapsed = reduced.matches ? duration : 0;
  let playing = !reduced.matches, dragging = false, visible = false, raf = 0, lastTime = null, positions;
  let connections = [], phase = -1;
  const point = (id,x,y) => { const [px,py,s]=positions[id]; return [px+x*s,py+y*s]; };
  function layout() {
    const small = mobile.matches;
    root.querySelector("svg.ma-canvas").setAttribute("viewBox", small ? "0 0 370 1634" : "0 0 1220 480");
    positions = small ? {
      video: [21,49,.8], audio: [207,96,.78], views: [74,334,1], model: [100,756,1],
      embeddings: [80,1080,1], distribution: [95,1384,1],
    } : {
      video: [24,66,.82], audio: [24,311,.82], views: [254,110,.92], model: [512,140,.9],
      embeddings: [718,128,1], distribution: [994,132,1],
    };
    Object.entries(positions).forEach(([id,[x,y,scale]]) => get(id).setAttribute("transform",`translate(${x} ${y}) scale(${scale})`));
    const specs = [];
    const add = (id,d,start,end,arrow=true) => specs.push({id,d,start,end,arrow});
    if (small) {
      add("source-video","M94 235V276H185",3500,4200,false);
      add("source-audio","M276 169V276H185",3500,4200,false);
      add("source-view","M185 276V282",4000,4400);
      views.forEach((_,i) => {
        const [x,y] = point("views",viewW,rowY(i)+40);
        add(`encode-${i}`,`M${x+2} ${y}H326V686H185V700`,7400+i*180,8150+i*180);
      });
      add("model-out","M185 980V1030",8900,10000);
      add("distribution","M185 1306V1324",13000,14000);
    } else {
      const [vx,vy] = point("video",194,98), [ax,ay] = point("audio",181,45);
      add("source-video",`M${vx+2} ${vy}H214`,3500,4300,false);
      add("source-audio",`M${ax+2} ${ay}H214`,3500,4300,false);
      add("source-join",`M214 ${Math.min(vy,point("views",0,40)[1])}V${ay}`,3500,4500,false);
      views.forEach((_,i) => {
        const [x,y] = point("views",0,rowY(i)+40);
        add(`view-in-${i}`,`M214 ${y}H${x-7}`,4100+i*180,4700+i*180);
        const [tx,ty] = point("views",viewW,rowY(i)+40), [mx,my] = point("model",0,50);
        add(`encode-${i}`,`M${tx+2} ${ty}H${tx+22}`,7400+i*180,7900+i*180,false);
      });
      const [bx,top] = point("views",viewW,40), [,bottom] = point("views",viewW,rowY(3)+40), [jx,jy] = point("model",0,50);
      add("encode-bus",`M${bx+22} ${top}V${bottom}`,7400,8300,false);
      add("encode-join",`M${bx+22} ${jy}H${jx-8}`,8000,8500);
      const [mx,my] = point("model",170,199), [ex] = point("embeddings",0,0);
      add("model-out",`M${mx+2} ${my}H${ex-9}`,8900,10000);
      add("distribution","M930 238H982",13000,14000);
    }
    get("connections").innerHTML = specs.map(edge => path(edge.d,`id="ma-path-${edge.id}" pathLength="1"`)).join("");
    connections = specs.map(edge => ({...edge,node:get(`path-${edge.id}`)}));
    render();
  }
  function render() {
    const t=elapsed;
    root.dataset.elapsed=Math.round(t);
    const convert=smooth(t,900,2550);
    get("mel-wipe").setAttribute("width",180*convert);
    opacity("wave",1-convert);opacity("scan",Math.sin(convert*Math.PI));
    get("scan").setAttribute("transform",`translate(${180*convert} 0)`);
    get("audio-label").textContent=convert<.5?"Waveform":"Spectrogram";
    const cut=smooth(t,2550,2950),separate=smooth(t,2950,3900);
    videoFrames.forEach(({sheet,tiles,grid})=>{
      sheet.style.display=t>=2950?"none":"";
      tiles.style.display=t>=2950?"":"none";
      tiles.style.setProperty("--patch-x",1-separate*.22);
      tiles.style.setProperty("--patch-y",1-separate*.22);
      grid.setAttribute("opacity",cut*(1-separate));
    });
    const depth=smooth(t,3100,3900);
    opacity("tubelets",depth);
    tubeletVolumes.forEach(({x,y,paths})=>{
      const dx=12*depth,dy=-14*depth,size=9.36;
      paths[0].setAttribute("d",`M${x} ${y}l${dx} ${dy}h${size}v${size}l${-dx} ${-dy}Z`);
      paths[1].setAttribute("d",`M${x} ${y}h${size}v${size}h-${size}Z M${x+size} ${y}l${dx} ${dy}`);
      paths[2].setAttribute("d",`M${x+dx} ${y+dy}l${dx} ${dy}h${size}v${size}l${-dx} ${-dy} M${x+dx+size} ${y+dy}l${dx} ${dy}`);
    });
    opacity("audio-grid",cut*(1-separate));
    opacity("audio-sheet",t>=2950?0:1);
    const tiles=get("audio-patches");
    tiles.style.display=t>=2950?"":"none";
    tiles.setAttribute("transform",`translate(90 45) scale(${1+separate*.04}) translate(-90 -45)`);
    tiles.style.setProperty("--patch-x",1-separate*.27);
    tiles.style.setProperty("--patch-y",1-separate*.20);
    opacity("views",smooth(t,4000,4650));
    views.forEach((view,i)=>{
      opacity(`view-${i}`,smooth(t,4000+i*170,4550+i*170));
      const drop=smooth(t,5200,6500),strength=.25+.65*smooth(t,6300,7500);
      opacity(`row-${i}-video`,strength*(view.video?1:1-drop));
      opacity(`row-${i}-audio`,strength*(view.audio?1:1-drop));
      const compact=smooth(t,6500,7500);
      if(!view.video||!view.audio) get(`row-${i}-${view.video?"video":"audio"}`).setAttribute("transform",`translate(${-60*compact} 0)`);
      if(!view.video||!view.audio)opacity(`drop-${i}`,drop*(1-compact));
    });
    opacity("model",smooth(t,7800,8450));opacity("readout",smooth(t,8500,9000));
    opacity("model-pulse",Math.max(Math.sin(smooth(t,8100,9550)*Math.PI)*.65,Math.sin(smooth(t,16200,17600)*Math.PI)*.8));
    opacity("projector-pulse",Math.sin(smooth(t,16200,17600)*Math.PI)*.8);
    opacity("embeddings",smooth(t,9300,10100));
    opacity("alignment",smooth(t,10500,11200));opacity("align-label",smooth(t,10500,11200));opacity("sigreg",smooth(t,13000,13700));
    opacity("distribution",smooth(t,13500,14100));
    connections.forEach(connection=>{
      const p=smooth(t,connection.start,connection.end);
      connection.node.style.strokeDasharray="1";
      connection.node.style.strokeDashoffset=1-p;
      connection.node.style.opacity=p;
      connection.node.style.markerEnd=p>.98&&connection.arrow?"url(#ma-arrow)":"none";
    });
    drawLosses(t);
    const nextPhase=stages.findLastIndex(([start])=>t>=start);
    if(phase!==nextPhase){
      phase=nextPhase;
      root.querySelectorAll("[data-time]").forEach(button=>button.setAttribute("aria-current",String(Number(button.dataset.phase)===phase)));
    }
    if (!dragging) get("scrubber").value=String(t);
    get("scrubber").style.setProperty("--progress",`${t/duration*100}%`);
    get("scrubber").setAttribute("aria-valuetext",`${(t/1000).toFixed(1)} seconds of 18. ${stages[phase][1]}.`);
  }
  function svgNode(tag, attributes, parent) {
    const node = document.createElementNS("http://www.w3.org/2000/svg",tag);
    Object.entries(attributes).forEach(([key,value]) => node.setAttribute(key,value));
    parent.append(node);
    return node;
  }
  const selected = [[-48,-42,ink,"G₁"],[48,42,ink,"G₂"],[-77,33,A,"A"],[75,-31,V,"V"]].map(([x,y,color,name]) => {
    const line = svgNode("line",{x2:125,y2:128,stroke:color,opacity:.5},get("align-lines"));
    const circle = svgNode("circle",{r:5.5,fill:color,stroke:"white","stroke-width":1.5},get("align-points"));
    return {x,y,line,circle};
  });
  const cloud = batch.map(([x,y,tx,ty]) => ({x,y,tx,ty,
      original:svgNode("circle",{r:2.6,fill:"#a1a1aa"},get("batch-original")),
      drop:svgNode("line",{},get("batch-drops")),
      projected:svgNode("circle",{r:2.1,fill:R},get("batch-projected")),
  }));
  const axis = svgNode("line",{},get("batch-axis"));
  const bars=Array.from({length:16},(_,i)=>svgNode("rect",{x:i*180/16+.8,width:180/16-1.6,rx:2,fill:"#64748b",opacity:.75},get("histogram")));
  const projectionSamples=batch.map(()=>svgNode("circle",{cy:30,r:2,fill:R,opacity:.6},get("projection-samples")));
  get("gaussian").setAttribute("d",Array.from({length:65},(_,i)=>`${i?"L":"M"}${i*180/64},${176-Math.exp(-.5*(-3+i*6/64)**2)/Math.sqrt(2*Math.PI)*195}`).join(" "));
  get("gaussian").style.strokeDasharray="1";
  const erf = x => { const s=Math.sign(x),a=Math.abs(x),k=1/(1+.3275911*a);
    return s*(1-(((((1.061405429*k-1.453152027)*k+1.421413741)*k-.284496736)*k+.254829592)*k)*Math.exp(-a*a)); };
  const cdf = x => .5*(1+erf(x/Math.SQRT2));
  const target=bars.map((_,i)=>(cdf(-3+(i+1)*6/16)-cdf(-3+i*6/16))/(6/16)*195);
  let lastLossTime = null;
  function drawLosses(t) {
    const time = Math.max(9300,Math.min(duration,t));
    if (time === lastLossTime) return;
    lastLossTime = time;
    const pull=1-.76*smooth(time,11100,12900);
    selected.forEach(({x,y,line,circle}) => {
      const px=125+x*pull,py=128+y*pull;
      line.setAttribute("x1",px);line.setAttribute("y1",py);
      circle.setAttribute("cx",px);circle.setAttribute("cy",py);
    });
    const project=smooth(time,13300,14300),angle=-.75+smooth(time,14300,15800)*1.05;
    const cos=Math.cos(angle),sin=Math.sin(angle),scalars=[];
    const settle=smooth(time,16000,17600);
    cloud.forEach(({x:bx,y:by,tx,ty,original,drop,projected}) => {
      const x=bx+(tx-bx)*settle,y=by+(ty-by)*settle,px=125+x*31,py=128+y*31;
      const scalar=x*cos+y*sin, ax=125+scalar*cos*31,ay=128+scalar*sin*31;
      original.setAttribute("cx",px);original.setAttribute("cy",py);
      drop.setAttribute("x1",px);drop.setAttribute("y1",py);
      drop.setAttribute("x2",ax);drop.setAttribute("y2",ay);
      projected.setAttribute("cx",px+(ax-px)*project);projected.setAttribute("cy",py+(ay-py)*project);
      scalars.push(scalar);
    });
    opacity("batch-drops",project);opacity("batch-projected",project*.7);opacity("batch-axis",smooth(time,13100,13900));
    Object.entries({x1:125-cos*102,y1:128-sin*102,x2:125+cos*102,y2:128+sin*102}).forEach(([key,value])=>axis.setAttribute(key,value));
    const bins=Array(16).fill(0);
    scalars.forEach((scalar,i)=>{
      projectionSamples[i].setAttribute("cx",Math.max(0,Math.min(180,(scalar+3)/6*180)));
      const bin=Math.max(0,Math.min(15,(scalar+3)/6*16-.5));
      const lo=Math.floor(bin),hi=Math.min(15,lo+1),fraction=bin-lo;
      bins[lo]+=1-fraction;bins[hi]+=fraction;
    });
    const rise=smooth(time,13800,14800);
    bars.forEach((bar,i)=>{
      const sample=bins[i]/batch.length/(6/16)*195;
      const height=(sample+(target[i]-sample)*settle)*rise;
      bar.setAttribute("y",176-height);bar.setAttribute("height",height);
    });
    get("gaussian").style.strokeDashoffset=1-smooth(time,14300,15400);
  }
  function stopFrame(){cancelAnimationFrame(raf);raf=0;lastTime=null;}
  function resume(){if(playing&&!dragging&&elapsed<duration&&visible&&!document.hidden&&!raf)raf=requestAnimationFrame(tick);}
  function tick(now){
    raf=0;
    if(!playing||dragging||!visible||document.hidden){lastTime=null;return;}
    if(lastTime!==null)elapsed=Math.min(duration,elapsed+now-lastTime);
    lastTime=elapsed>=duration?null:now;
    render();resume();
  }
  function seek(value){stopFrame();elapsed=Math.max(0,Math.min(duration,value));render();resume();}
  function finishScrub(){
    if(!dragging)return;
    dragging=false;lastTime=null;render();resume();
  }
  get("scrubber").addEventListener("pointerdown",()=>{dragging=true;stopFrame();});
  get("scrubber").addEventListener("input",event=>seek(Number(event.target.value)));
  window.addEventListener("pointerup",finishScrub);
  window.addEventListener("pointercancel",finishScrub);
  window.addEventListener("blur",finishScrub);
  root.querySelectorAll("[data-time]").forEach(button=>button.addEventListener("click",()=>seek(Number(button.dataset.time))));
  new IntersectionObserver(entries=>{
    visible=entries[entries.length-1].isIntersecting;if(visible)resume();else stopFrame();
  },{threshold:0}).observe(root);
  document.addEventListener("visibilitychange",()=>{if(document.hidden){dragging=false;stopFrame();}else resume();});
  mobile.addEventListener("change",layout);
  reduced.addEventListener("change",()=>{if(reduced.matches){playing=false;dragging=false;seek(duration);}});
  layout();
})();
