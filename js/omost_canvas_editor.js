import { app } from "../../scripts/app.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { ComfyApp } from "../../scripts/app.js";
import { ClipspaceDialog } from "../../extensions/core/clipspace.js";


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
            }, [
                $el("div", {}, "hello world"),
                ...this.createButtons(),
            ]),
        ]);
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

    show() {
        this.element.style.display = "flex";
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
                        ComfyApp.copyToClipspace(this);
                        ComfyApp.clipspace_return_node = this;

                        let dlg = OmostCanvasDialog.getInstance();
                        dlg.show();
                    },
                });
            });
        }
    }
});
