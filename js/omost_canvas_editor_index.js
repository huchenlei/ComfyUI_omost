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
            id: "comfyui-openpose-editor",
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

    close() {
        const targetNode = ComfyApp.clipspace_return_node;
        const textAreaElement = targetNode.widgets[0].element;
        textAreaElement.value = this.textAreaElement.value;
        super.close();
    }

    show() {
        if (!this.is_layout_created) {
            this.createLayout();
            this.is_layout_created = true;
        }

        const targetNode = ComfyApp.clipspace_return_node;
        const textAreaElement = targetNode.widgets[0].element;
        this.setCanvasJSONString(textAreaElement.value);

        this.element.style.display = "flex";
    }

    createLayout() {
        this.textAreaElement = $el("textarea", {
            style: {
                width: "100%",
                height: "100%",
            },
        });

        this.element.appendChild(this.textAreaElement);
    }

    setCanvasJSONString(jsonString) {
        this.textAreaElement.value = jsonString;
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
