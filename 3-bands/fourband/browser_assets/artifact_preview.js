import { safeLocalURL } from "./workspace_core.mjs";

export const isPreviewImage = (artifact) =>
  ["image/png", "image/jpeg", "image/webp", "image/gif"].includes(artifact.content_type);

export class ArtifactPreview {
  constructor(base) {
    this.base = base;
    this.images = [];
    this.scope = "";
    this.index = -1;
    this.dialog = document.getElementById("artifact-preview");
    this.$ = (name) => document.getElementById(`artifact-${name}`);
    this.$("close").onclick = () => this.close();
    this.$("prev").onclick = () => this.move(-1);
    this.$("next").onclick = () => this.move(1);
    this.$("zoom-in").onclick = () => this.zoom(1.25);
    this.$("zoom-out").onclick = () => this.zoom(0.8);
    this.$("fit").onclick = () => { this.scale = 1; this.resize(); };
    this.dialog.addEventListener("cancel", (event) => {
      event.preventDefault();
      this.close();
    });
    this.dialog.addEventListener("keydown", (event) => {
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        this.move(event.key === "ArrowLeft" ? -1 : 1);
      }
    });
    new ResizeObserver(() => this.resize()).observe(this.$("viewport"));
  }

  setArtifacts(artifacts, scope) {
    const images = artifacts.filter(isPreviewImage);
    const changed = scope !== this.scope || JSON.stringify(images) !== JSON.stringify(this.images);
    if (changed) this.close();
    this.scope = scope;
    this.images = images;
  }

  open(id, opener) {
    const index = this.images.findIndex((image) => image.id === id);
    if (index < 0) return;
    this.opener = opener;
    if (!this.dialog.open) this.dialog.showModal();
    this.select(index);
    this.$("close").focus();
  }

  close() {
    if (this.dialog.open) this.dialog.close();
    this.image = null;
    this.$("viewport").replaceChildren();
    this.$("download").removeAttribute("href");
    this.$("download").removeAttribute("download");
    if (this.opener?.isConnected) this.opener.focus();
    this.opener = null;
    this.index = -1;
  }

  move(delta) {
    const index = this.index + delta;
    if (index >= 0 && index < this.images.length) this.select(index);
  }

  select(index) {
    const artifact = this.images[index];
    const href = safeLocalURL(`api/workspace/artifact/${encodeURIComponent(artifact.id)}`, this.base);
    this.index = index;
    this.scale = 1;
    this.$("title").textContent = artifact.label ?? artifact.id;
    this.$("count").textContent = `${index + 1} / ${this.images.length}`;
    this.$("prev").disabled = index === 0;
    this.$("next").disabled = index === this.images.length - 1;
    if (!this.dialog.contains(document.activeElement) || document.activeElement.disabled)
      this.$("close").focus();
    this.$("download").href = href;
    this.$("download").download = artifact.label ?? "";
    this.$("image-status").textContent = "加载中…";
    for (const name of ["zoom-in", "zoom-out", "fit"]) this.$(name).disabled = true;
    this.$("zoom").textContent = "";
    const img = document.createElement("img");
    this.image = img;
    img.id = "artifact-image";
    img.alt = artifact.label ?? "导出图像";
    img.hidden = true;
    img.onload = () => {
      if (this.image !== img) return;
      img.hidden = false;
      this.$("image-status").textContent = "";
      this.$("fit").disabled = false;
      this.resize();
    };
    img.onerror = () => {
      if (this.image !== img) return;
      this.$("image-status").textContent = "图像加载失败，请关闭后重试。";
    };
    this.$("viewport").replaceChildren(img);
    img.src = href;
  }

  zoom(factor) {
    this.scale = Math.min(8, Math.max(0.25, this.scale * factor));
    this.resize();
  }

  resize() {
    const img = this.image, viewport = this.$("viewport");
    if (!this.dialog.open || !img?.naturalWidth || img.hidden) return;
    const fit = Math.min(1, (viewport.clientWidth - 24) / img.naturalWidth,
      (viewport.clientHeight - 24) / img.naturalHeight);
    img.style.width = `${Math.max(1, img.naturalWidth * fit * this.scale)}px`;
    img.style.height = `${Math.max(1, img.naturalHeight * fit * this.scale)}px`;
    this.$("zoom").textContent = `${Math.round(fit * this.scale * 100)}%`;
    this.$("zoom-in").disabled = this.scale >= 8;
    this.$("zoom-out").disabled = this.scale <= 0.25;
    if (this.scale === 1) viewport.scrollTo(0, 0);
  }
}
