import * as THREE from "./vendor/three/three.module.min.js";
import { OrbitControls } from "./vendor/three/OrbitControls.js";

export class BrainView {
  constructor(host, labelsHost, onSelect, onError) {
    this.host = host;
    this.labelsHost = labelsHost;
    this.onSelect = onSelect;
    this.onError = onError;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#f4f6f7");
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 10000);
    this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.setAttribute(
      "aria-label",
      "Interactive brain anatomy",
    );
    this.renderer.domElement.setAttribute("role", "img");
    this.host.prepend(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.screenSpacePanning = true;
    this.controls.addEventListener("change", () => this.render());
    this.scene.add(new THREE.HemisphereLight("#ffffff", "#7d858c", 2.1));
    const light = new THREE.DirectionalLight("#ffffff", 2.4);
    light.position.set(-300, -200, 350);
    this.scene.add(light);
    const fill = new THREE.DirectionalLight("#d7e5ec", 1.2);
    fill.position.set(200, 100, -80);
    this.scene.add(fill);
    this.cortex = [];
    this.contacts = [];
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.sphere = new THREE.SphereGeometry(1, 16, 12);
    this.style = {
      brain_color: "#c8ccd0",
      brain_opacity: 0.65,
      point_size: 14,
      electrodes: true,
      labels: false,
      class_filter: "all",
      channel: "",
      colors: {},
    };
    this.renderer.domElement.addEventListener("pointerdown", (event) => {
      this.down = [event.clientX, event.clientY];
    });
    this.renderer.domElement.addEventListener("pointerup", (event) => {
      if (
        !this.down ||
        Math.hypot(event.clientX - this.down[0], event.clientY - this.down[1]) >
          5
      )
        return;
      const rect = this.renderer.domElement.getBoundingClientRect();
      this.pointer.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        (-(event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const hits = this.raycaster.intersectObjects(
        this.contacts
          .filter((item) => item.mesh.visible)
          .map((item) => item.mesh),
      );
      if (!hits.length) return;
      const hit = hits[0];
      if (this.style.brain_opacity >= 0.995) {
        const cortexHit = this.raycaster.intersectObjects(this.cortex)[0];
        if (cortexHit && cortexHit.distance < hit.distance) return;
      }
      this.onSelect(hit.object.userData.channel);
    });
    this.renderer.domElement.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      this.onError(
        "WebGL context lost. Reload the page to restore anatomy. Curves remain available.",
      );
    });
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.resize();
  }

  load(payload) {
    const warnings = [...(payload.warnings || [])];
    const bounds = new THREE.Box3();
    for (const mesh of payload.meshes || []) {
      if (
        !mesh.positions?.length ||
        mesh.positions.length % 3 ||
        !mesh.positions.every(Number.isFinite) ||
        !mesh.indices?.length ||
        mesh.indices.length % 3 ||
        !mesh.indices.every(
          (value) =>
            Number.isInteger(value) &&
            value >= 0 &&
            value < mesh.positions.length / 3,
        )
      ) {
        warnings.push(
          `Invalid anatomy geometry: ${mesh.name || "unnamed mesh"}`,
        );
        continue;
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(mesh.positions, 3),
      );
      geometry.setIndex(mesh.indices);
      geometry.computeVertexNormals();
      geometry.computeBoundingBox();
      bounds.union(geometry.boundingBox);
      const material = new THREE.MeshStandardMaterial({
        color: this.style.brain_color,
        roughness: 0.86,
        metalness: 0,
        side: THREE.DoubleSide,
      });
      const surface = new THREE.Mesh(geometry, material);
      surface.name = mesh.name || "Cortex";
      this.scene.add(surface);
      this.cortex.push(surface);
    }
    for (const electrode of payload.electrodes || []) {
      if (
        !Array.isArray(electrode.position) ||
        electrode.position.length !== 3 ||
        !electrode.position.every(Number.isFinite)
      ) {
        warnings.push(`Unmapped channel: ${electrode.channel}`);
        continue;
      }
      const material = new THREE.MeshStandardMaterial({
        color: "#167d91",
        roughness: 0.45,
        metalness: 0,
      });
      const mesh = new THREE.Mesh(this.sphere, material);
      // Source scanner/tkRAS coordinates remain untouched; only camera and glyph scale change.
      mesh.position.fromArray(electrode.position);
      mesh.userData.channel = electrode.channel;
      this.scene.add(mesh);
      const label = document.createElement("span");
      label.className = "electrode-label";
      label.textContent = electrode.channel;
      label.hidden = true;
      this.labelsHost.append(label);
      this.contacts.push({
        mesh,
        label,
        channel: electrode.channel,
        cluster: electrode.label,
      });
      if (this.cortex.length === 0) bounds.expandByPoint(mesh.position);
    }
    if (this.cortex.length === 0)
      warnings.push("No valid cortical mesh is available.");
    if (bounds.isEmpty())
      throw new Error(
        "No valid anatomy or electrode coordinates are available.",
      );
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const radius = Math.max(size.length() / 2, 1);
    const names = (payload.meshes || [])
      .map((mesh) => mesh.name)
      .join(" ")
      .toLowerCase();
    let axis;
    if (
      /(^|[\s_./-])lh([\s_./-]|$)|left/.test(names) &&
      !/(^|[\s_./-])rh([\s_./-]|$)|right/.test(names)
    )
      axis = new THREE.Vector3(-1, 0, 0.1);
    else if (
      /(^|[\s_./-])rh([\s_./-]|$)|right/.test(names) &&
      !/(^|[\s_./-])lh([\s_./-]|$)|left/.test(names)
    )
      axis = new THREE.Vector3(1, 0, 0.1);
    else {
      const average = new THREE.Vector3();
      for (const item of this.contacts) average.add(item.mesh.position);
      if (this.contacts.length)
        average.divideScalar(this.contacts.length).sub(center);
      axis =
        Math.abs(average.x) > Math.abs(average.y)
          ? new THREE.Vector3(Math.sign(average.x) || -1, 0, 0.1)
          : new THREE.Vector3(0, Math.sign(average.y) || -1, 0.1);
    }
    this.camera.near = Math.max(radius / 2000, 0.01);
    this.camera.far = radius * 100;
    this.controls.minDistance = radius * 0.15;
    this.controls.maxDistance = radius * 20;
    this.controls.target.copy(center);
    const aspect = Math.min(this.camera.aspect, 1);
    const distance =
      ((radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2))) *
        1.05) /
      aspect;
    this.camera.position
      .copy(center)
      .add(axis.normalize().multiplyScalar(distance));
    this.camera.lookAt(center);
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.initialCamera = this.cameraState();
    this.update(this.style);
    return {
      warnings,
      mapped: this.contacts.length,
      meshes: this.cortex.length,
    };
  }

  resize() {
    const width = this.host.clientWidth,
      height = this.host.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.render();
  }

  update(style) {
    this.style = { ...this.style, ...style };
    for (const mesh of this.cortex) {
      mesh.material.color.set(this.style.brain_color);
      mesh.material.opacity = this.style.brain_opacity;
      mesh.material.transparent = this.style.brain_opacity < 0.995;
      mesh.material.depthWrite = !mesh.material.transparent;
      mesh.visible = this.style.brain_opacity > 0;
      mesh.material.needsUpdate = true;
    }
    for (const item of this.contacts) {
      item.mesh.visible =
        this.style.electrodes &&
        (this.style.class_filter === "all" ||
          String(item.cluster) === this.style.class_filter);
      item.mesh.material.color.set(
        this.style.colors[item.cluster] || "#167d91",
      );
      item.mesh.material.emissive.set(
        item.channel === this.style.channel ? "#13383d" : "#000000",
      );
      item.label.classList.toggle(
        "selected",
        item.channel === this.style.channel,
      );
    }
    this.render();
  }

  render() {
    if (!this.renderer || !this.host.clientHeight) return;
    this.camera.updateMatrixWorld();
    const direction = new THREE.Vector3();
    this.camera.getWorldDirection(direction);
    const projected = new THREE.Vector3();
    for (const item of this.contacts) {
      const depth = Math.abs(
        item.mesh.position.clone().sub(this.camera.position).dot(direction),
      );
      const worldPerPixel =
        (2 * depth * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) /
        this.host.clientHeight;
      item.mesh.scale.setScalar(
        Math.max((worldPerPixel * this.style.point_size) / 2, 0.01) *
          (item.channel === this.style.channel ? 1.15 : 1),
      );
      projected.copy(item.mesh.position).project(this.camera);
      item.label.hidden =
        !this.style.labels ||
        !item.mesh.visible ||
        projected.z < -1 ||
        projected.z > 1 ||
        Math.abs(projected.x) > 1 ||
        Math.abs(projected.y) > 1;
      if (!item.label.hidden && this.style.brain_opacity >= 0.995) {
        this.raycaster.setFromCamera(
          new THREE.Vector2(projected.x, projected.y),
          this.camera,
        );
        const hit = this.raycaster.intersectObjects(this.cortex)[0];
        if (
          hit &&
          hit.distance <
            this.camera.position.distanceTo(item.mesh.position) -
              item.mesh.scale.x
        )
          item.label.hidden = true;
      }
      item.label.style.left = `${((projected.x + 1) * this.host.clientWidth) / 2}px`;
      item.label.style.top = `${((1 - projected.y) * this.host.clientHeight) / 2}px`;
    }
    this.renderer.render(this.scene, this.camera);
  }

  cameraState() {
    return {
      position: this.camera.position.toArray(),
      target: this.controls.target.toArray(),
      up: this.camera.up.toArray(),
      zoom: this.camera.zoom,
    };
  }

  restoreCamera(value) {
    if (
      !value ||
      !["position", "target", "up"].every(
        (key) =>
          Array.isArray(value[key]) &&
          value[key].length === 3 &&
          value[key].every(Number.isFinite),
      )
    )
      return;
    this.camera.position.fromArray(value.position);
    this.camera.up.fromArray(value.up);
    this.controls.target.fromArray(value.target);
    this.camera.zoom =
      Number.isFinite(value.zoom) && value.zoom > 0 ? value.zoom : 1;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.render();
  }

  reset() {
    this.restoreCamera(this.initialCamera);
  }

  exportPNG() {
    this.render();
    const canvas = document.createElement("canvas");
    canvas.width = this.renderer.domElement.width;
    canvas.height = this.renderer.domElement.height;
    const context = canvas.getContext("2d");
    context.drawImage(this.renderer.domElement, 0, 0);
    const ratio = canvas.width / this.host.clientWidth;
    context.scale(ratio, ratio);
    context.font = "10px Segoe UI, Arial";
    for (const item of this.contacts)
      if (!item.label.hidden) {
        const text = item.channel,
          width = context.measureText(text).width;
        const x = parseFloat(item.label.style.left) - width / 2,
          y = parseFloat(item.label.style.top) + 8;
        context.fillStyle = "rgba(255,255,255,0.85)";
        context.fillRect(x - 3, y - 2, width + 6, 14);
        context.fillStyle = "#263943";
        context.fillText(text, x, y + 9);
      }
    return canvas.toDataURL("image/png");
  }
}
