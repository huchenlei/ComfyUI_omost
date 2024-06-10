import { app } from "../../scripts/app.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { ComfyApp } from "../../scripts/app.js";


function addMenuHandler(nodeType, cb) {
    const getOpts = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function () {
        const r = getOpts.apply(this, arguments);
        cb.apply(this, arguments);
        return r;
    };
}

class OmostCanvasDialog extends ComfyDialog {
    static timeout = 5000;
    static instance = null;

    static getInstance() {
        if (!OmostCanvasDialog.instance) {
            OmostCanvasDialog.instance = new OmostCanvasDialog();
        }

        return OmostCanvasDialog.instance;
    }

    constructor() {
        super();
        this.element = $el("div.comfy-modal", {
            parent: document.body,
            style: {
                width: "80vw",
                height: "80vh",
            },
        }, [
            $el("div.comfy-modal-content", {
                style: {
                    width: "100%",
                    height: "100%",
                },
            }, this.createButtons()),
        ]);
        this.is_layout_created = false;
    }

    createButtons() {
        const closeBtn = $el("button", {
            type: "button",
            textContent: "Close",
            onclick: () => this.close(),
        });
        return [
            closeBtn,
        ];
    }

    async close() {
        const targetNode = ComfyApp.clipspace_return_node;
        const textAreaElement = targetNode.widgets[0].element;
        textAreaElement.value = await this.getCanvasJSONString();
        super.close();
    }

    async show() {
        if (!this.is_layout_created) {
            this.createLayout();
            this.is_layout_created = true;
            await this.waitIframeReady();
        }

        const targetNode = ComfyApp.clipspace_return_node;
        const textAreaElement = targetNode.widgets[0].element;
        this.element.style.display = "flex";
        this.setCanvasJSONString(textAreaElement.value);
    }

    createLayout() {
        this.iframeElement = $el("iframe", {
            // Change to http://localhost:5174 for local dev
            src: "https://huchenlei.github.io/omost_region_editor",
            style: {
                width: "100%",
                height: "100%",
                border: "none",
            },
        });
        const modalContent = this.element.querySelector(".comfy-modal-content");
        while (modalContent.firstChild) {
            modalContent.removeChild(modalContent.firstChild);
        }
        modalContent.appendChild(this.iframeElement);
        modalContent.appendChild(...this.createButtons());
    }

    waitIframeReady() {
        return new Promise((resolve, reject) => {
            window.addEventListener("message", (event) => {
                if (event.source !== this.iframeElement.contentWindow) {
                    return;
                }
                if (event.data.type === "ready") {
                    resolve();
                }
            });
            setTimeout(() => {
                reject(new Error("Timeout"));
            }, OmostCanvasDialog.timeout);
        });
    }

    getCanvasJSONString() {
        return new Promise((resolve, reject) => {
            window.addEventListener("message", (event) => {
                if (event.source !== this.iframeElement.contentWindow) {
                    return;
                }
                if (event.data.type === "save") {
                    resolve(JSON.stringify(event.data.regions));
                }
            });

            this.iframeElement.contentWindow.postMessage({ type: "save" }, "*");

            setTimeout(() => {
                reject(new Error("Timeout"));
            }, OmostCanvasDialog.timeout);
        });
    }

    setCanvasJSONString(jsonString) {
        this.iframeElement.contentWindow.postMessage(
            { type: "update", regions: JSON.parse(jsonString) }, "*");
    }
}

function isOmostLoadCanvasConditioningNode(nodeData) {
    return nodeData.name === "OmostLoadCanvasConditioningNode";
}

app.registerExtension({
    name: "huchenlei.EditOmostCanvas",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (isOmostLoadCanvasConditioningNode(nodeData)) {
            addMenuHandler(nodeType, function (_, options) {
                options.unshift({
                    content: "Open in Omost Canvas Editor",
                    callback: () => {
                        // `this` is the node instance
                        ComfyApp.copyToClipspace(this);
                        ComfyApp.clipspace_return_node = this;

                        const dlg = OmostCanvasDialog.getInstance();
                        dlg.show();
                    },
                });
            });
        }
    }
});
