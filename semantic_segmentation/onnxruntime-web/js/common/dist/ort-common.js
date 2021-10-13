/*!
 * ONNX Runtime Common v1.9.0
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
(function webpackUniversalModuleDefinition(root, factory) {
	if(typeof exports === 'object' && typeof module === 'object')
		module.exports = factory();
	else if(typeof define === 'function' && define.amd)
		define([], factory);
	else if(typeof exports === 'object')
		exports["ort"] = factory();
	else
		root["ort"] = factory();
})(self, function() {
return /******/ (function() { // webpackBootstrap
/******/ 	"use strict";
/******/ 	var __webpack_modules__ = ({

/***/ "./lib/backend-impl.ts":
/*!*****************************!*\
  !*** ./lib/backend-impl.ts ***!
  \*****************************/
/***/ (function(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "registerBackend": function() { return /* binding */ registerBackend; },
/* harmony export */   "resolveBackend": function() { return /* binding */ resolveBackend; }
/* harmony export */ });
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
var __awaiter = (undefined && undefined.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (undefined && undefined.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __values = (undefined && undefined.__values) || function(o) {
    var s = typeof Symbol === "function" && Symbol.iterator, m = s && o[s], i = 0;
    if (m) return m.call(o);
    if (o && typeof o.length === "number") return {
        next: function () {
            if (o && i >= o.length) o = void 0;
            return { value: o && o[i++], done: !o };
        }
    };
    throw new TypeError(s ? "Object is not iterable." : "Symbol.iterator is not defined.");
};
var backends = {};
var backendsSortedByPriority = [];
/**
 * Register a backend.
 *
 * @param name - the name as a key to lookup as an execution provider.
 * @param backend - the backend object.
 * @param priority - an integer indicating the priority of the backend. Higher number means higher priority.
 */
var registerBackend = function (name, backend, priority) {
    if (backend && typeof backend.init === 'function' && typeof backend.createSessionHandler === 'function') {
        var currentBackend = backends[name];
        if (currentBackend === undefined) {
            backends[name] = { backend: backend, priority: priority };
        }
        else if (currentBackend.backend === backend) {
            return;
        }
        else {
            throw new Error("backend \"" + name + "\" is already registered");
        }
        for (var i = 0; i < backendsSortedByPriority.length; i++) {
            if (backends[backendsSortedByPriority[i]].priority <= priority) {
                backendsSortedByPriority.splice(i, 0, name);
                return;
            }
        }
        backendsSortedByPriority.push(name);
        return;
    }
    throw new TypeError('not a valid backend');
};
/**
 * Resolve backend by specified hints.
 *
 * @param backendHints - a list of execution provider names to lookup. If omitted use registered backends as list.
 * @returns a promise that resolves to the backend.
 */
var resolveBackend = function (backendHints) { return __awaiter(void 0, void 0, void 0, function () {
    var backendNames, errors, backendNames_1, backendNames_1_1, backendName, backendInfo, e_1, e_2_1;
    var e_2, _a;
    return __generator(this, function (_b) {
        switch (_b.label) {
            case 0:
                backendNames = backendHints.length === 0 ? backendsSortedByPriority : backendHints;
                errors = [];
                _b.label = 1;
            case 1:
                _b.trys.push([1, 9, 10, 11]);
                backendNames_1 = __values(backendNames), backendNames_1_1 = backendNames_1.next();
                _b.label = 2;
            case 2:
                if (!!backendNames_1_1.done) return [3 /*break*/, 8];
                backendName = backendNames_1_1.value;
                backendInfo = backends[backendName];
                if (!backendInfo) return [3 /*break*/, 7];
                if (backendInfo.initialized) {
                    return [2 /*return*/, backendInfo.backend];
                }
                else if (backendInfo.initializing) {
                    throw new Error("backend \"" + backendName + "\" is being initialized; cannot initialize multiple times.");
                }
                else if (backendInfo.aborted) {
                    return [3 /*break*/, 7]; // current backend is unavailable; try next
                }
                _b.label = 3;
            case 3:
                _b.trys.push([3, 5, 6, 7]);
                backendInfo.initializing = true;
                return [4 /*yield*/, backendInfo.backend.init()];
            case 4:
                _b.sent();
                backendInfo.initialized = true;
                return [2 /*return*/, backendInfo.backend];
            case 5:
                e_1 = _b.sent();
                errors.push({ name: backendName, err: e_1 });
                backendInfo.aborted = true;
                return [3 /*break*/, 7];
            case 6:
                backendInfo.initializing = false;
                return [7 /*endfinally*/];
            case 7:
                backendNames_1_1 = backendNames_1.next();
                return [3 /*break*/, 2];
            case 8: return [3 /*break*/, 11];
            case 9:
                e_2_1 = _b.sent();
                e_2 = { error: e_2_1 };
                return [3 /*break*/, 11];
            case 10:
                try {
                    if (backendNames_1_1 && !backendNames_1_1.done && (_a = backendNames_1.return)) _a.call(backendNames_1);
                }
                finally { if (e_2) throw e_2.error; }
                return [7 /*endfinally*/];
            case 11: throw new Error("no available backend found. ERR: " + errors.map(function (e) { return "[" + e.name + "] " + e.err; }).join(', '));
        }
    });
}); };


/***/ }),

/***/ "./lib/backend.ts":
/*!************************!*\
  !*** ./lib/backend.ts ***!
  \************************/
/***/ (function(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "registerBackend": function() { return /* reexport safe */ _backend_impl__WEBPACK_IMPORTED_MODULE_0__.registerBackend; }
/* harmony export */ });
/* harmony import */ var _backend_impl__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./backend-impl */ "./lib/backend-impl.ts");
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.



/***/ }),

/***/ "./lib/env-impl.ts":
/*!*************************!*\
  !*** ./lib/env-impl.ts ***!
  \*************************/
/***/ (function(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "EnvImpl": function() { return /* binding */ EnvImpl; }
/* harmony export */ });
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
var EnvImpl = /** @class */ (function () {
    function EnvImpl() {
        this.wasm = {};
        this.webgl = {};
        this.logLevelInternal = 'warning';
    }
    Object.defineProperty(EnvImpl.prototype, "logLevel", {
        get: function () {
            return this.logLevelInternal;
        },
        // TODO standadize the getter and setter convention in env for other fields.
        set: function (value) {
            if (value === undefined) {
                return;
            }
            if (typeof value !== 'string' || ['verbose', 'info', 'warning', 'error', 'fatal'].indexOf(value) === -1) {
                throw new Error("Unsupported logging level: " + value);
            }
            this.logLevelInternal = value;
        },
        enumerable: false,
        configurable: true
    });
    return EnvImpl;
}());



/***/ }),

/***/ "./lib/env.ts":
/*!********************!*\
  !*** ./lib/env.ts ***!
  \********************/
/***/ (function(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "env": function() { return /* binding */ env; }
/* harmony export */ });
/* harmony import */ var _env_impl__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./env-impl */ "./lib/env-impl.ts");
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * Represent a set of flags as a global singleton.
 */
var env = new _env_impl__WEBPACK_IMPORTED_MODULE_0__.EnvImpl();


/***/ }),

/***/ "./lib/inference-session-impl.ts":
/*!***************************************!*\
  !*** ./lib/inference-session-impl.ts ***!
  \***************************************/
/***/ (function(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "InferenceSession": function() { return /* binding */ InferenceSession; }
/* harmony export */ });
/* harmony import */ var _backend_impl__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./backend-impl */ "./lib/backend-impl.ts");
/* harmony import */ var _tensor__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./tensor */ "./lib/tensor.ts");
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
var __awaiter = (undefined && undefined.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (undefined && undefined.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var __values = (undefined && undefined.__values) || function(o) {
    var s = typeof Symbol === "function" && Symbol.iterator, m = s && o[s], i = 0;
    if (m) return m.call(o);
    if (o && typeof o.length === "number") return {
        next: function () {
            if (o && i >= o.length) o = void 0;
            return { value: o && o[i++], done: !o };
        }
    };
    throw new TypeError(s ? "Object is not iterable." : "Symbol.iterator is not defined.");
};


var InferenceSession = /** @class */ (function () {
    function InferenceSession(handler) {
        this.handler = handler;
    }
    InferenceSession.prototype.run = function (feeds, arg1, arg2) {
        return __awaiter(this, void 0, void 0, function () {
            var fetches, options, isFetchesEmpty, arg1_1, arg1_1_1, name, isFetches, arg1Keys, _a, _b, name, v, _c, _d, name, _e, _f, name, results, returnValue, key;
            var e_1, _g, e_2, _h, e_3, _j, e_4, _k;
            return __generator(this, function (_l) {
                switch (_l.label) {
                    case 0:
                        fetches = {};
                        options = {};
                        // check inputs
                        if (typeof feeds !== 'object' || feeds === null || feeds instanceof _tensor__WEBPACK_IMPORTED_MODULE_1__.Tensor || Array.isArray(feeds)) {
                            throw new TypeError('\'feeds\' must be an object that use input names as keys and OnnxValue as corresponding values.');
                        }
                        isFetchesEmpty = true;
                        // determine which override is being used
                        if (typeof arg1 === 'object') {
                            if (arg1 === null) {
                                throw new TypeError('Unexpected argument[1]: cannot be null.');
                            }
                            if (arg1 instanceof _tensor__WEBPACK_IMPORTED_MODULE_1__.Tensor) {
                                throw new TypeError('\'fetches\' cannot be a Tensor');
                            }
                            if (Array.isArray(arg1)) {
                                if (arg1.length === 0) {
                                    throw new TypeError('\'fetches\' cannot be an empty array.');
                                }
                                isFetchesEmpty = false;
                                try {
                                    // output names
                                    for (arg1_1 = __values(arg1), arg1_1_1 = arg1_1.next(); !arg1_1_1.done; arg1_1_1 = arg1_1.next()) {
                                        name = arg1_1_1.value;
                                        if (typeof name !== 'string') {
                                            throw new TypeError('\'fetches\' must be a string array or an object.');
                                        }
                                        if (this.outputNames.indexOf(name) === -1) {
                                            throw new RangeError("'fetches' contains invalid output name: " + name + ".");
                                        }
                                        fetches[name] = null;
                                    }
                                }
                                catch (e_1_1) { e_1 = { error: e_1_1 }; }
                                finally {
                                    try {
                                        if (arg1_1_1 && !arg1_1_1.done && (_g = arg1_1.return)) _g.call(arg1_1);
                                    }
                                    finally { if (e_1) throw e_1.error; }
                                }
                                if (typeof arg2 === 'object' && arg2 !== null) {
                                    options = arg2;
                                }
                                else if (typeof arg2 !== 'undefined') {
                                    throw new TypeError('\'options\' must be an object.');
                                }
                            }
                            else {
                                isFetches = false;
                                arg1Keys = Object.getOwnPropertyNames(arg1);
                                try {
                                    for (_a = __values(this.outputNames), _b = _a.next(); !_b.done; _b = _a.next()) {
                                        name = _b.value;
                                        if (arg1Keys.indexOf(name) !== -1) {
                                            v = arg1[name];
                                            if (v === null || v instanceof _tensor__WEBPACK_IMPORTED_MODULE_1__.Tensor) {
                                                isFetches = true;
                                                isFetchesEmpty = false;
                                                fetches[name] = v;
                                            }
                                        }
                                    }
                                }
                                catch (e_2_1) { e_2 = { error: e_2_1 }; }
                                finally {
                                    try {
                                        if (_b && !_b.done && (_h = _a.return)) _h.call(_a);
                                    }
                                    finally { if (e_2) throw e_2.error; }
                                }
                                if (isFetches) {
                                    if (typeof arg2 === 'object' && arg2 !== null) {
                                        options = arg2;
                                    }
                                    else if (typeof arg2 !== 'undefined') {
                                        throw new TypeError('\'options\' must be an object.');
                                    }
                                }
                                else {
                                    options = arg1;
                                }
                            }
                        }
                        else if (typeof arg1 !== 'undefined') {
                            throw new TypeError('Unexpected argument[1]: must be \'fetches\' or \'options\'.');
                        }
                        try {
                            // check if all inputs are in feed
                            for (_c = __values(this.inputNames), _d = _c.next(); !_d.done; _d = _c.next()) {
                                name = _d.value;
                                if (typeof feeds[name] === 'undefined') {
                                    throw new Error("input '" + name + "' is missing in 'feeds'.");
                                }
                            }
                        }
                        catch (e_3_1) { e_3 = { error: e_3_1 }; }
                        finally {
                            try {
                                if (_d && !_d.done && (_j = _c.return)) _j.call(_c);
                            }
                            finally { if (e_3) throw e_3.error; }
                        }
                        // if no fetches is specified, we use the full output names list
                        if (isFetchesEmpty) {
                            try {
                                for (_e = __values(this.outputNames), _f = _e.next(); !_f.done; _f = _e.next()) {
                                    name = _f.value;
                                    fetches[name] = null;
                                }
                            }
                            catch (e_4_1) { e_4 = { error: e_4_1 }; }
                            finally {
                                try {
                                    if (_f && !_f.done && (_k = _e.return)) _k.call(_e);
                                }
                                finally { if (e_4) throw e_4.error; }
                            }
                        }
                        return [4 /*yield*/, this.handler.run(feeds, fetches, options)];
                    case 1:
                        results = _l.sent();
                        returnValue = {};
                        for (key in results) {
                            if (Object.hasOwnProperty.call(results, key)) {
                                returnValue[key] = new _tensor__WEBPACK_IMPORTED_MODULE_1__.Tensor(results[key].type, results[key].data, results[key].dims);
                            }
                        }
                        return [2 /*return*/, returnValue];
                }
            });
        });
    };
    InferenceSession.create = function (arg0, arg1, arg2, arg3) {
        return __awaiter(this, void 0, void 0, function () {
            var filePathOrUint8Array, options, buffer, byteOffset, byteLength, eps, backendHints, backend, handler;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        options = {};
                        if (typeof arg0 === 'string') {
                            filePathOrUint8Array = arg0;
                            if (typeof arg1 === 'object' && arg1 !== null) {
                                options = arg1;
                            }
                            else if (typeof arg1 !== 'undefined') {
                                throw new TypeError('\'options\' must be an object.');
                            }
                        }
                        else if (arg0 instanceof Uint8Array) {
                            filePathOrUint8Array = arg0;
                            if (typeof arg1 === 'object' && arg1 !== null) {
                                options = arg1;
                            }
                            else if (typeof arg1 !== 'undefined') {
                                throw new TypeError('\'options\' must be an object.');
                            }
                        }
                        else if (arg0 instanceof ArrayBuffer ||
                            (typeof SharedArrayBuffer !== 'undefined' && arg0 instanceof SharedArrayBuffer)) {
                            buffer = arg0;
                            byteOffset = 0;
                            byteLength = arg0.byteLength;
                            if (typeof arg1 === 'object' && arg1 !== null) {
                                options = arg1;
                            }
                            else if (typeof arg1 === 'number') {
                                byteOffset = arg1;
                                if (!Number.isSafeInteger(byteOffset)) {
                                    throw new RangeError('\'byteOffset\' must be an integer.');
                                }
                                if (byteOffset < 0 || byteOffset >= buffer.byteLength) {
                                    throw new RangeError("'byteOffset' is out of range [0, " + buffer.byteLength + ").");
                                }
                                byteLength = arg0.byteLength - byteOffset;
                                if (typeof arg2 === 'number') {
                                    byteLength = arg2;
                                    if (!Number.isSafeInteger(byteLength)) {
                                        throw new RangeError('\'byteLength\' must be an integer.');
                                    }
                                    if (byteLength <= 0 || byteOffset + byteLength > buffer.byteLength) {
                                        throw new RangeError("'byteLength' is out of range (0, " + (buffer.byteLength - byteOffset) + "].");
                                    }
                                    if (typeof arg3 === 'object' && arg3 !== null) {
                                        options = arg3;
                                    }
                                    else if (typeof arg3 !== 'undefined') {
                                        throw new TypeError('\'options\' must be an object.');
                                    }
                                }
                                else if (typeof arg2 !== 'undefined') {
                                    throw new TypeError('\'byteLength\' must be a number.');
                                }
                            }
                            else if (typeof arg1 !== 'undefined') {
                                throw new TypeError('\'options\' must be an object.');
                            }
                            filePathOrUint8Array = new Uint8Array(buffer, byteOffset, byteLength);
                        }
                        else {
                            throw new TypeError('Unexpected argument[0]: must be \'path\' or \'buffer\'.');
                        }
                        eps = options.executionProviders || [];
                        backendHints = eps.map(function (i) { return typeof i === 'string' ? i : i.name; });
                        return [4 /*yield*/, (0,_backend_impl__WEBPACK_IMPORTED_MODULE_0__.resolveBackend)(backendHints)];
                    case 1:
                        backend = _a.sent();
                        return [4 /*yield*/, backend.createSessionHandler(filePathOrUint8Array, options)];
                    case 2:
                        handler = _a.sent();
                        return [2 /*return*/, new InferenceSession(handler)];
                }
            });
        });
    };
    InferenceSession.prototype.startProfiling = function () {
        this.handler.startProfiling();
    };
    InferenceSession.prototype.endProfiling = function () {
        this.handler.endProfiling();
    };
    Object.defineProperty(InferenceSession.prototype, "inputNames", {
        get: function () {
            return this.handler.inputNames;
        },
        enumerable: false,
        configurable: true
    });
    Object.defineProperty(InferenceSession.prototype, "outputNames", {
        get: function () {
            return this.handler.outputNames;
        },
        enumerable: false,
        configurable: true
    });
    return InferenceSession;
}());



/***/ }),

/***/ "./lib/inference-session.ts":
/*!**********************************!*\
  !*** ./lib/inference-session.ts ***!
  \**********************************/
/***/ (function(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "InferenceSession": function() { return /* binding */ InferenceSession; }
/* harmony export */ });
/* harmony import */ var _inference_session_impl__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./inference-session-impl */ "./lib/inference-session-impl.ts");
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// eslint-disable-next-line @typescript-eslint/naming-convention
var InferenceSession = _inference_session_impl__WEBPACK_IMPORTED_MODULE_0__.InferenceSession;


/***/ }),

/***/ "./lib/onnx-value.ts":
/*!***************************!*\
  !*** ./lib/onnx-value.ts ***!
  \***************************/
/***/ (function(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.



/***/ }),

/***/ "./lib/tensor-impl.ts":
/*!****************************!*\
  !*** ./lib/tensor-impl.ts ***!
  \****************************/
/***/ (function(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "Tensor": function() { return /* binding */ Tensor; }
/* harmony export */ });
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
var isBigInt64ArrayAvailable = typeof BigInt64Array !== 'undefined' && typeof BigInt64Array.from === 'function';
var isBigUint64ArrayAvailable = typeof BigUint64Array !== 'undefined' && typeof BigUint64Array.from === 'function';
// a runtime map that maps type string to TypedArray constructor. Should match Tensor.DataTypeMap.
var NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP = new Map([
    ['float32', Float32Array],
    ['uint8', Uint8Array],
    ['int8', Int8Array],
    ['uint16', Uint16Array],
    ['int16', Int16Array],
    ['int32', Int32Array],
    ['bool', Uint8Array],
    ['float64', Float64Array],
    ['uint32', Uint32Array],
]);
// a runtime map that maps type string to TypedArray constructor. Should match Tensor.DataTypeMap.
var NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP = new Map([
    [Float32Array, 'float32'],
    [Uint8Array, 'uint8'],
    [Int8Array, 'int8'],
    [Uint16Array, 'uint16'],
    [Int16Array, 'int16'],
    [Int32Array, 'int32'],
    [Float64Array, 'float64'],
    [Uint32Array, 'uint32'],
]);
if (isBigInt64ArrayAvailable) {
    NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.set('int64', BigInt64Array);
    NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP.set(BigInt64Array, 'int64');
}
if (isBigUint64ArrayAvailable) {
    NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.set('uint64', BigUint64Array);
    NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP.set(BigUint64Array, 'uint64');
}
/**
 * calculate size from dims.
 *
 * @param dims the dims array. May be an illegal input.
 */
var calculateSize = function (dims) {
    var size = 1;
    for (var i = 0; i < dims.length; i++) {
        var dim = dims[i];
        if (typeof dim !== 'number' || !Number.isSafeInteger(dim)) {
            throw new TypeError("dims[" + i + "] must be an integer, got: " + dim);
        }
        if (dim < 0) {
            throw new RangeError("dims[" + i + "] must be a non-negative integer, got: " + dim);
        }
        size *= dim;
    }
    return size;
};
var Tensor = /** @class */ (function () {
    function Tensor(arg0, arg1, arg2) {
        var type;
        var data;
        var dims;
        // check whether arg0 is type or data
        if (typeof arg0 === 'string') {
            //
            // Override: constructor(type, data, ...)
            //
            type = arg0;
            dims = arg2;
            if (arg0 === 'string') {
                // string tensor
                if (!Array.isArray(arg1)) {
                    throw new TypeError('A string tensor\'s data must be a string array.');
                }
                // we don't check whether every element in the array is string; this is too slow. we assume it's correct and
                // error will be populated at inference
                data = arg1;
            }
            else {
                // numeric tensor
                var typedArrayConstructor = NUMERIC_TENSOR_TYPE_TO_TYPEDARRAY_MAP.get(arg0);
                if (typedArrayConstructor === undefined) {
                    throw new TypeError("Unsupported tensor type: " + arg0 + ".");
                }
                if (Array.isArray(arg1)) {
                    // use 'as any' here because TypeScript's check on type of 'SupportedTypedArrayConstructors.from()' produces
                    // incorrect results.
                    // 'typedArrayConstructor' should be one of the typed array prototype objects.
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    data = typedArrayConstructor.from(arg1);
                }
                else if (arg1 instanceof typedArrayConstructor) {
                    data = arg1;
                }
                else {
                    throw new TypeError("A " + type + " tensor's data must be type of " + typedArrayConstructor);
                }
            }
        }
        else {
            //
            // Override: constructor(data, ...)
            //
            dims = arg1;
            if (Array.isArray(arg0)) {
                // only boolean[] and string[] is supported
                if (arg0.length === 0) {
                    throw new TypeError('Tensor type cannot be inferred from an empty array.');
                }
                var firstElementType = typeof arg0[0];
                if (firstElementType === 'string') {
                    type = 'string';
                    data = arg0;
                }
                else if (firstElementType === 'boolean') {
                    type = 'bool';
                    // 'arg0' is of type 'boolean[]'. Uint8Array.from(boolean[]) actually works, but typescript thinks this is
                    // wrong type. We use 'as any' to make it happy.
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    data = Uint8Array.from(arg0);
                }
                else {
                    throw new TypeError("Invalid element type of data array: " + firstElementType + ".");
                }
            }
            else {
                // get tensor type from TypedArray
                var mappedType = NUMERIC_TENSOR_TYPEDARRAY_TO_TYPE_MAP.get(arg0.constructor);
                if (mappedType === undefined) {
                    throw new TypeError("Unsupported type for tensor data: " + arg0.constructor + ".");
                }
                type = mappedType;
                data = arg0;
            }
        }
        // type and data is processed, now processing dims
        if (dims === undefined) {
            // assume 1-D tensor if dims omitted
            dims = [data.length];
        }
        else if (!Array.isArray(dims)) {
            throw new TypeError('A tensor\'s dims must be a number array');
        }
        // perform check
        var size = calculateSize(dims);
        if (size !== data.length) {
            throw new Error("Tensor's size(" + size + ") does not match data length(" + data.length + ").");
        }
        this.dims = dims;
        this.type = type;
        this.data = data;
        this.size = size;
    }
    //#endregion
    //#region tensor utilities
    Tensor.prototype.reshape = function (dims) {
        return new Tensor(this.type, this.data, dims);
    };
    return Tensor;
}());



/***/ }),

/***/ "./lib/tensor.ts":
/*!***********************!*\
  !*** ./lib/tensor.ts ***!
  \***********************/
/***/ (function(__unused_webpack_module, __webpack_exports__, __webpack_require__) {

__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "Tensor": function() { return /* binding */ Tensor; }
/* harmony export */ });
/* harmony import */ var _tensor_impl__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./tensor-impl */ "./lib/tensor-impl.ts");
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// eslint-disable-next-line @typescript-eslint/naming-convention
var Tensor = _tensor_impl__WEBPACK_IMPORTED_MODULE_0__.Tensor;


/***/ })

/******/ 	});
/************************************************************************/
/******/ 	// The module cache
/******/ 	var __webpack_module_cache__ = {};
/******/ 	
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/ 		// Check if module is in cache
/******/ 		var cachedModule = __webpack_module_cache__[moduleId];
/******/ 		if (cachedModule !== undefined) {
/******/ 			return cachedModule.exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = __webpack_module_cache__[moduleId] = {
/******/ 			// no module.id needed
/******/ 			// no module.loaded needed
/******/ 			exports: {}
/******/ 		};
/******/ 	
/******/ 		// Execute the module function
/******/ 		__webpack_modules__[moduleId](module, module.exports, __webpack_require__);
/******/ 	
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/ 	
/************************************************************************/
/******/ 	/* webpack/runtime/define property getters */
/******/ 	!function() {
/******/ 		// define getter functions for harmony exports
/******/ 		__webpack_require__.d = function(exports, definition) {
/******/ 			for(var key in definition) {
/******/ 				if(__webpack_require__.o(definition, key) && !__webpack_require__.o(exports, key)) {
/******/ 					Object.defineProperty(exports, key, { enumerable: true, get: definition[key] });
/******/ 				}
/******/ 			}
/******/ 		};
/******/ 	}();
/******/ 	
/******/ 	/* webpack/runtime/hasOwnProperty shorthand */
/******/ 	!function() {
/******/ 		__webpack_require__.o = function(obj, prop) { return Object.prototype.hasOwnProperty.call(obj, prop); }
/******/ 	}();
/******/ 	
/******/ 	/* webpack/runtime/make namespace object */
/******/ 	!function() {
/******/ 		// define __esModule on exports
/******/ 		__webpack_require__.r = function(exports) {
/******/ 			if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 				Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 			}
/******/ 			Object.defineProperty(exports, '__esModule', { value: true });
/******/ 		};
/******/ 	}();
/******/ 	
/************************************************************************/
var __webpack_exports__ = {};
// This entry need to be wrapped in an IIFE because it need to be isolated against other modules in the chunk.
!function() {
/*!**********************!*\
  !*** ./lib/index.ts ***!
  \**********************/
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "registerBackend": function() { return /* reexport safe */ _backend__WEBPACK_IMPORTED_MODULE_0__.registerBackend; },
/* harmony export */   "env": function() { return /* reexport safe */ _env__WEBPACK_IMPORTED_MODULE_1__.env; },
/* harmony export */   "InferenceSession": function() { return /* reexport safe */ _inference_session__WEBPACK_IMPORTED_MODULE_2__.InferenceSession; },
/* harmony export */   "Tensor": function() { return /* reexport safe */ _tensor__WEBPACK_IMPORTED_MODULE_3__.Tensor; }
/* harmony export */ });
/* harmony import */ var _backend__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ./backend */ "./lib/backend.ts");
/* harmony import */ var _env__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./env */ "./lib/env.ts");
/* harmony import */ var _inference_session__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./inference-session */ "./lib/inference-session.ts");
/* harmony import */ var _tensor__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./tensor */ "./lib/tensor.ts");
/* harmony import */ var _onnx_value__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./onnx-value */ "./lib/onnx-value.ts");
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.






}();
/******/ 	return __webpack_exports__;
/******/ })()
;
});
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9vcnQvd2VicGFjay91bml2ZXJzYWxNb2R1bGVEZWZpbml0aW9uIiwid2VicGFjazovL29ydC8uL2xpYi9iYWNrZW5kLWltcGwudHMiLCJ3ZWJwYWNrOi8vb3J0Ly4vbGliL2JhY2tlbmQudHMiLCJ3ZWJwYWNrOi8vb3J0Ly4vbGliL2Vudi1pbXBsLnRzIiwid2VicGFjazovL29ydC8uL2xpYi9lbnYudHMiLCJ3ZWJwYWNrOi8vb3J0Ly4vbGliL2luZmVyZW5jZS1zZXNzaW9uLWltcGwudHMiLCJ3ZWJwYWNrOi8vb3J0Ly4vbGliL2luZmVyZW5jZS1zZXNzaW9uLnRzIiwid2VicGFjazovL29ydC8uL2xpYi9vbm54LXZhbHVlLnRzIiwid2VicGFjazovL29ydC8uL2xpYi90ZW5zb3ItaW1wbC50cyIsIndlYnBhY2s6Ly9vcnQvLi9saWIvdGVuc29yLnRzIiwid2VicGFjazovL29ydC93ZWJwYWNrL2Jvb3RzdHJhcCIsIndlYnBhY2s6Ly9vcnQvd2VicGFjay9ydW50aW1lL2RlZmluZSBwcm9wZXJ0eSBnZXR0ZXJzIiwid2VicGFjazovL29ydC93ZWJwYWNrL3J1bnRpbWUvaGFzT3duUHJvcGVydHkgc2hvcnRoYW5kIiwid2VicGFjazovL29ydC93ZWJwYWNrL3J1bnRpbWUvbWFrZSBuYW1lc3BhY2Ugb2JqZWN0Iiwid2VicGFjazovL29ydC8uL2xpYi9pbmRleC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7OztBQUFBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBLENBQUM7QUFDRCxPOzs7Ozs7Ozs7Ozs7Ozs7QUNWQSw0REFBNEQ7QUFDNUQsa0NBQWtDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFhbEMsSUFBTSxRQUFRLEdBQWtDLEVBQUUsQ0FBQztBQUNuRCxJQUFNLHdCQUF3QixHQUFhLEVBQUUsQ0FBQztBQUU5Qzs7Ozs7O0dBTUc7QUFDSSxJ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Q2pGO1dBQ0E7V0FDQSxFOzs7OztXQ1BBLDZDQUE2Qyx3REFBd0QsRTs7Ozs7V0NBckc7V0FDQTtXQUNBO1dBQ0Esc0RBQXNELGtCQUFrQjtXQUN4RTtXQUNBLCtDQUErQyxjQUFjO1dBQzdELEU7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUNOQSw0REFBNEQ7QUFDNUQsa0NBQWtDO0FBRVI7QUFDSjtBQUNjO0FBQ1g7QUFDSSIsImZpbGUiOiJvcnQtY29tbW9uLmpzIiwic291cmNlc0NvbnRlbnQiOlsiKGZ1bmN0aW9uIHdlYnBhY2tVbml2ZXJzYWxNb2R1bGVEZWZpbml0aW9uKHJvb3QsIGZhY3RvcnkpIHtcblx0aWYodHlwZW9mIGV4cG9ydHMgPT09ICdvYmplY3QnICYmIHR5cGVvZiBtb2R1bGUgPT09ICdvYmplY3QnKVxuXHRcdG1vZHVsZS5leHBvcnRzID0gZmFjdG9yeSgpO1xuXHRlbHNlIGlmKHR5cGVvZiBkZWZpbmUgPT09ICdmdW5jdGlvbicgJiYgZGVmaW5lLmFtZClcblx0XHRkZWZpbmUoW10sIGZhY3RvcnkpO1xuXHRlbHNlIGlmKHR5cGVvZiBleHBvcnRzID09PSAnb2JqZWN0Jylcblx0XHRleHBvcnRzW1wib3J0XCJdID0gZmFjdG9yeSgpO1xuXHRlbHNlXG5cdFx0cm9vdFtcIm9ydFwiXSA9IGZhY3RvcnkoKTtcbn0pKHNlbGYsIGZ1bmN0aW9uKCkge1xucmV0dXJuICIsIi8vIENvcHlyaWdodCAoYykgTWljcm9zb2Z0IENvcnBvcmF0aW9uLiBBbGwgcmlnaHRzIHJlc2VydmVkLlxuLy8gTGljZW5zZWQgdW5kZXIgdGhlIE1JVCBMaWNlbnNlLlxuXG5pbXBvcnQge0JhY2tlbmR9IGZyb20gJy4vYmFja2VuZCc7XG5cbmludGVyZmFjZSBCYWNrZW5kSW5mbyB7XG4gIGJhY2tlbmQ6IEJhY2tlbmQ7XG4gIHByaW9yaXR5OiBudW1iZXI7XG5cbiAgaW5pdGlhbGl6aW5nPzogYm9vbGVhbjtcbiAgaW5pdGlhbGl6ZWQ/OiBib29sZWFuO1xuICBhYm9ydGVkPzogYm9vbGVhbjtcbn1cblxuY29uc3QgYmFja2VuZHM6IHtbbmFtZTogc3RyaW5nXTogQmFja2VuZEluZm99ID0ge307XG5jb25zdCBiYWNrZW5kc1NvcnRlZEJ5UHJpb3JpdHk6IHN0cmluZ1tdID0gW107XG5cbi8qKlxuICogUmVnaXN0ZXIgYSBiYWNrZW5kLlxuICpcbiAqIEBwYXJhbSBuYW1lIC0gdGhlIG5hbWUgYXMgYSBrZXkgdG8gbG9va3VwIGFzIGFuIGV4ZWN1dGlvbiBwcm92aWRlci5cbiAqIEBwYXJhbSBiYWNrZW5kIC0gdGhlIGJhY2tlbmQgb2JqZWN0LlxuICogQHBhcmFtIHByaW9yaXR5IC0gYW4gaW50ZWdlciBpbmRpY2F0aW5nIHRoZSBwcmlvcml0eSBvZiB0aGUgYmFja2VuZC4gSGlnaGVyIG51bWJlciBtZWFucyBoaWdoZXIgcHJpb3JpdHkuXG4gKi9cbmV4cG9ydCBjb25zdCByZWdpc3RlckJhY2tlbmQgPSAobmFtZTogc3RyaW5nLCBiYWNrZW5kOiBCYWNrZW5kLCBwcmlvcml0eTogbnVtYmVyKTogdm9pZCA9PiB7XG4gIGlmIChiYWNrZW5kICYmIHR5cGVvZiBiYWNrZW5kLmluaXQgPT09ICdmdW5jdGlvbicgJiYgdHlwZW9mIGJhY2tlbmQuY3JlYXRlU2Vzc2lvbkhhbmRsZXIgPT09ICdmdW5jdGlvbicpIHtcbiAgICBjb25zdCBjdXJyZW50QmFja2VuZCA9IGJhY2tlbmRzW25hbWVdO1xuICAgIGlmIChjdXJyZW50QmFja2VuZCA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICBiYWNrZW5kc1tuYW1lXSA9IHtiYWNrZW5kLCBwcmlvcml0eX07XG4gICAgfSBlbHNlIGlmIChjdXJyZW50QmFja2VuZC5iYWNrZW5kID09PSBiYWNrZW5kKSB7XG4gICAgICByZXR1cm47XG4gICAgfSBlbHNlIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgYmFja2VuZCBcIiR7bmFtZX1cIiBpcyBhbHJlYWR5IHJlZ2lzdGVyZWRgKTtcbiAgICB9XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGJhY2tlbmRzU29ydGVkQnlQcmlvcml0eS5sZW5ndGg7IGkrKykge1xuICAgICAgaWYgKGJhY2tlbmRzW2JhY2tlbmRzU29ydGVkQnlQcmlvcml0eVtpXV0ucHJpb3JpdHkgPD0gcHJpb3JpdHkpIHtcbiAgICAgICAgYmFja2VuZHNTb3J0ZWRCeVByaW9yaXR5LnNwbGljZShpLCAwLCBuYW1lKTtcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuICAgIH1cbiAgICBiYWNrZW5kc1NvcnRlZEJ5UHJpb3JpdHkucHVzaChuYW1lKTtcbiAgICByZXR1cm47XG4gIH1cblxuICB0aHJvdyBuZXcgVHlwZUVycm9yKCdub3QgYSB2YWxpZCBiYWNrZW5kJyk7XG59O1xuXG4vKipcbiAqIFJlc29sdmUgYmFja2VuZCBieSBzcGVjaWZpZWQgaGludHMuXG4gKlxuICogQHBhcmFtIGJhY2tlbmRIaW50cyAtIGEgbGlzdCBvZiBleGVjdXRpb24gcHJvdmlkZXIgbmFtZXMgdG8gbG9va3VwLiBJZiBvbWl0dGVkIHVzZSByZWdpc3RlcmVkIGJhY2tlbmRzIGFzIGxpc3QuXG4gKiBAcmV0dXJucyBhIHByb21pc2UgdGhhdCByZXNvbHZlcyB0byB0aGUgYmFja2VuZC5cbiAqL1xuZXhwb3J0IGNvbnN0IHJlc29sdmVCYWNrZW5kID0gYXN5bmMoYmFja2VuZEhpbnRzOiByZWFkb25seSBzdHJpbmdbXSk6IFByb21pc2U8QmFja2VuZD4gPT4ge1xuICBjb25zdCBiYWNrZW5kTmFtZXMgPSBiYWNrZW5kSGludHMubGVuZ3RoID09PSAwID8gYmFja2VuZHNTb3J0ZWRCeVByaW9yaXR5IDogYmFja2VuZEhpbnRzO1xuICBjb25zdCBlcnJvcnMgPSBbXTtcbiAgZm9yIChjb25zdCBiYWNrZW5kTmFtZSBvZiBiYWNrZW5kTmFtZXMpIHtcbiAgICBjb25zdCBiYWNrZW5kSW5mbyA9IGJhY2tlbmRzW2JhY2tlbmROYW1lXTtcbiAgICBpZiAoYmFja2VuZEluZm8pIHtcbiAgICAgIGlmIChiYWNrZW5kSW5mby5pbml0aWFsaXplZCkge1xuICAgICAgICByZXR1cm4gYmFja2VuZEluZm8uYmFja2VuZDtcbiAgICAgIH0gZWxzZSBpZiAoYmFja2VuZEluZm8uaW5pdGlhbGl6aW5nKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihgYmFja2VuZCBcIiR7YmFja2VuZE5hbWV9XCIgaXMgYmVpbmcgaW5pdGlhbGl6ZWQ7IGNhbm5vdCBpbml0aWFsaXplIG11bHRpcGxlIHRpbWVzLmApO1xuICAgICAgfSBlbHNlIGlmIChiYWNrZW5kSW5mby5hYm9ydGVkKSB7XG4gICAgICAgIGNvbnRpbnVlOyAgLy8gY3VycmVudCBiYWNrZW5kIGlzIHVuYXZhaWxhYmxlOyB0cnkgbmV4dFxuICAgICAgfVxuXG4gICAgICB0cnkge1xuICAgICAgICBiYWNrZW5kSW5mby5pbml0aWFsaXppbmcgPSB0cnVlO1xuICAgICAgICBhd2FpdCBiYWNrZW5kSW5mby5iYWNrZW5kLmluaXQoKTtcbiAgICAgICAgYmFja2VuZEluZm8uaW5pdGlhbGl6ZWQgPSB0cnVlO1xuICAgICAgICByZXR1cm4gYmFja2VuZEluZm8uYmFja2VuZDtcbiAgICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAgICAgZXJyb3JzLnB1c2goe25hbWU6IGJhY2tlbmROYW1lLCBlcnI6IGV9KTtcbiAgICAgICAgYmFja2VuZEluZm8uYWJvcnRlZCA9IHRydWU7XG4gICAgICB9IGZpbmFsbHkge1xuICAgICAgICBiYWNrZW5kSW5mby5pbml0aWFsaXppbmcgPSBmYWxzZTtcbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICB0aHJvdyBuZXcgRXJyb3IoYG5vIGF2YWlsYWJsZSBiYWNrZW5kIGZvdW5kLiBFUlI6ICR7ZXJyb3JzLm1hcChlID0+IGBbJHtlLm5hbWV9XSAke2UuZXJyfWApLmpvaW4oJywgJyl9YCk7XG59O1xuIiwiLy8gQ29weXJpZ2h0IChjKSBNaWNyb3NvZnQgQ29ycG9yYXRpb24uIEFsbCByaWdodHMgcmVzZXJ2ZWQuXG4vLyBMaWNlbnNlZCB1bmRlciB0aGUgTUlUIExpY2Vuc2UuXG5cbmltcG9ydCB7SW5mZXJlbmNlU2Vzc2lvbn0gZnJvbSAnLi9pbmZlcmVuY2Utc2Vzc2lvbic7XG5pbXBvcnQge09ubnhWYWx1ZX0gZnJvbSAnLi9vbm54LXZhbHVlJztcblxuZXhwb3J0IGRlY2xhcmUgbmFtZXNwYWNlIFNlc3Npb25IYW5kbGVyIHtcbiAgdHlwZSBGZWVkc1R5cGUgPSB7W25hbWU6IHN0cmluZ106IE9ubnhWYWx1ZX07XG4gIHR5cGUgRmV0Y2hlc1R5cGUgPSB7W25hbWU6IHN0cmluZ106IE9ubnhWYWx1ZSB8IG51bGx9O1xuICB0eXBlIFJldHVyblR5cGUgPSB7W25hbWU6IHN0cmluZ106IE9ubnhWYWx1ZX07XG59XG5cbi8qKlxuICogUmVwcmVzZW50IGEgaGFuZGxlciBpbnN0YW5jZSBvZiBhbiBpbmZlcmVuY2Ugc2Vzc2lvbi5cbiAqL1xuZXhwb3J0IGludGVyZmFjZSBTZXNzaW9uSGFuZGxlciB7XG4gIGRpc3Bvc2UoKTogUHJvbWlzZTx2b2lkPjtcblxuICByZWFkb25seSBpbnB1dE5hbWVzOiByZWFkb25seSBzdHJpbmdbXTtcbiAgcmVhZG9ubHkgb3V0cHV0TmFtZXM6IHJlYWRvbmx5IHN0cmluZ1tdO1xuXG4gIHN0YXJ0UHJvZmlsaW5nKCk6IHZvaWQ7XG4gIGVuZFByb2ZpbGluZygpOiB2b2lkO1xuXG4gIHJ1bihmZWVkczogU2Vzc2lvbkhhbmRsZXIuRmVlZHNUeXBlLCBmZXRjaGVzOiBTZXNzaW9uSGFuZGxlci5GZXRjaGVzVHlwZSxcbiAgICAgIG9wdGlvbnM6IEluZmVyZW5jZVNlc3Npb24uUnVuT3B0aW9ucyk6IFByb21pc2U8U2Vzc2lvbkhhbmRsZXIuUmV0dXJuVHlwZT47XG59XG5cbi8qKlxuICogUmVwcmVzZW50IGEgYmFja2VuZCB0aGF0IHByb3ZpZGVzIGltcGxlbWVudGF0aW9uIG9mIG1vZGVsIGluZmVyZW5jaW5nLlxuICovXG5leHBvcnQgaW50ZXJmYWNlIEJhY2tlbmQge1xuICAvKipcbiAgICogSW5pdGlhbGl6ZSB0aGUgYmFja2VuZCBhc3luY2hyb25vdXNseS4gU2hvdWxkIHRocm93IHdoZW4gZmFpbGVkLlxuICAgKi9cbiAgaW5pdCgpOiBQcm9taXNlPHZvaWQ+O1xuXG4gIGNyZWF0ZVNlc3Npb25IYW5kbGVyKHVyaU9yQnVmZmVyOiBzdHJpbmd8VWludDhBcnJheSwgb3B0aW9ucz86IEluZmVyZW5jZVNlc3Npb24uU2Vzc2lvbk9wdGlvbnMpOlxuICAgICAgUHJvbWlzZTxTZXNzaW9uSGFuZGxlcj47XG59XG5cbmV4cG9ydCB7cmVnaXN0ZXJCYWNrZW5kfSBmcm9tICcuL2JhY2tlbmQtaW1wbCc7XG4iLCIvLyBDb3B5cmlnaHQgKGMpIE1pY3Jvc29mdCBDb3Jwb3JhdGlvbi4gQWxsIHJpZ2h0cyByZXNlcnZlZC5cbi8vIExpY2Vuc2VkIHVuZGVyIHRoZSBNSVQgTGljZW5zZS5cblxuaW1wb3J0IHtFbnZ9IGZyb20gJy4vZW52JztcblxudHlwZSBMb2dMZXZlbFR5cGUgPSBFbnZbJ2xvZ0xldmVsJ107XG5leHBvcnQgY2xhc3MgRW52SW1wbCBpbXBsZW1lbnRzIEVudiB7XG4gIGNvbnN0cnVjdG9yKCkge1xuICAgIHRoaXMud2FzbSA9IHt9O1xuICAgIHRoaXMud2ViZ2wgPSB7fTtcbiAgICB0aGlzLmxvZ0xldmVsSW50ZXJuYWwgPSAnd2FybmluZyc7XG4gIH1cblxuICAvLyBUT0RPIHN0YW5kYWRpemUgdGhlIGdldHRlciBhbmQgc2V0dGVyIGNvbnZlbnRpb24gaW4gZW52IGZvciBvdGhlciBmaWVsZHMuXG4gIHNldCBsb2dMZXZlbCh2YWx1ZTogTG9nTGV2ZWxUeXBlKSB7XG4gICAgaWYgKHZhbHVlID09PSB1bmRlZmluZWQpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgaWYgKHR5cGVvZiB2YWx1ZSAhPT0gJ3N0cmluZycgfHwgWyd2ZXJib3NlJywgJ2luZm8nLCAnd2FybmluZycsICdlcnJvcicsICdmYXRhbCddLmluZGV4T2YodmFsdWUpID09PSAtMSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBVbnN1cHBvcnRlZCBsb2dnaW5nIGxldmVsOiAke3ZhbHVlfWApO1xuICAgIH1cbiAgICB0aGlzLmxvZ0xldmVsSW50ZXJuYWwgPSB2YWx1ZTtcbiAgfVxuICBnZXQgbG9nTGV2ZWwoKTogTG9nTGV2ZWxUeXBlIHtcbiAgICByZXR1cm4gdGhpcy5sb2dMZXZlbEludGVybmFsO1xuICB9XG5cbiAgZGVidWc/OiBib29sZWFuO1xuXG4gIHdhc206IEVudi5XZWJBc3NlbWJseUZsYWdzO1xuXG4gIHdlYmdsOiBFbnYuV2ViR0xGbGFncztcblxuICBbbmFtZTogc3RyaW5nXTogdW5rbm93bjtcblxuICBwcml2YXRlIGxvZ0xldmVsSW50ZXJuYWw6IFJlcXVpcmVkPExvZ0xldmVsVHlwZT47XG59XG4iLCIvLyBDb3B5cmlnaHQgKGMpIE1pY3Jvc29mdCBDb3Jwb3JhdGlvbi4gQWxsIHJpZ2h0cyByZXNlcnZlZC5cbi8vIExpY2Vuc2VkIHVuZGVyIHRoZSBNSVQgTGljZW5zZS5cblxuaW1wb3J0IHtFbnZJbXBsfSBmcm9tICcuL2Vudi1pbXBsJztcbmV4cG9ydCBkZWNsYXJlIG5hbWVzcGFjZSBFbnYge1xuICBleHBvcnQgdHlwZSBXYXNtUHJlZml4T3JGaWxlUGF0aHMgPSBzdHJpbmd8e1xuICAgICdvcnQtd2FzbS53YXNtJz86IHN0cmluZztcbiAgICAnb3J0LXdhc20tdGhyZWFkZWQud2FzbSc/OiBzdHJpbmc7XG4gICAgJ29ydC13YXNtLXNpbWQud2FzbSc/OiBzdHJpbmc7XG4gICAgJ29ydC13YXNtLXNpbWQtdGhyZWFkZWQud2FzbSc/OiBzdHJpbmc7XG4gIH07XG4gIGV4cG9ydCBpbnRlcmZhY2UgV2ViQXNzZW1ibHlGbGFncyB7XG4gICAgLyoqXG4gICAgICogc2V0IG9yIGdldCBudW1iZXIgb2YgdGhyZWFkKHMpLiBJZiBvbWl0dGVkIG9yIHNldCB0byAwLCBudW1iZXIgb2YgdGhyZWFkKHMpIHdpbGwgYmUgZGV0ZXJtaW5lZCBieSBzeXN0ZW0uIElmIHNldFxuICAgICAqIHRvIDEsIG5vIHdvcmtlciB0aHJlYWQgd2lsbCBiZSBzcGF3bmVkLlxuICAgICAqXG4gICAgICogVGhpcyBzZXR0aW5nIGlzIGF2YWlsYWJsZSBvbmx5IHdoZW4gV2ViQXNzZW1ibHkgbXVsdGl0aHJlYWQgZmVhdHVyZSBpcyBhdmFpbGFibGUgaW4gY3VycmVudCBjb250ZXh0LlxuICAgICAqL1xuICAgIG51bVRocmVhZHM/OiBudW1iZXI7XG5cbiAgICAvKipcbiAgICAgKiBzZXQgb3IgZ2V0IGEgYm9vbGVhbiB2YWx1ZSBpbmRpY2F0aW5nIHdoZXRoZXIgdG8gZW5hYmxlIFNJTUQuIElmIHNldCB0byBmYWxzZSwgU0lNRCB3aWxsIGJlIGZvcmNlbHkgZGlzYWJsZWQuXG4gICAgICpcbiAgICAgKiBUaGlzIHNldHRpbmcgaXMgYXZhaWxhYmxlIG9ubHkgd2hlbiBXZWJBc3NlbWJseSBTSU1EIGZlYXR1cmUgaXMgYXZhaWxhYmxlIGluIGN1cnJlbnQgY29udGV4dC5cbiAgICAgKi9cbiAgICBzaW1kPzogYm9vbGVhbjtcblxuICAgIC8qKlxuICAgICAqIFNldCBvciBnZXQgYSBudW1iZXIgc3BlY2lmeWluZyB0aGUgdGltZW91dCBmb3IgaW5pdGlhbGl6YXRpb24gb2YgV2ViQXNzZW1ibHkgYmFja2VuZCwgaW4gbWlsbGlzZWNvbmRzLiBBIHplcm9cbiAgICAgKiB2YWx1ZSBpbmRpY2F0ZXMgbm8gdGltZW91dCBpcyBzZXQuIChkZWZhdWx0IGlzIDApXG4gICAgICovXG4gICAgaW5pdFRpbWVvdXQ/OiBudW1iZXI7XG5cbiAgICAvKipcbiAgICAgKiBTZXQgYSBjdXN0b20gVVJMIHByZWZpeCB0byB0aGUgLndhc20gZmlsZXMgb3IgYSBzZXQgb2Ygb3ZlcnJpZGVzIGZvciBlYWNoIC53YXNtIGZpbGUuIFRoZSBvdmVycmlkZSBwYXRoIHNob3VsZCBiZVxuICAgICAqIGFuIGFic29sdXRlIHBhdGguXG4gICAgICovXG4gICAgd2FzbVBhdGhzPzogV2FzbVByZWZpeE9yRmlsZVBhdGhzO1xuXG4gICAgLyoqXG4gICAgICogU2V0IG9yIGdldCBhIGJvb2xlYW4gdmFsdWUgaW5kaWNhdGluZyB3aGV0aGVyIHRvIHByb3h5IHRoZSBleGVjdXRpb24gb2YgbWFpbiB0aHJlYWQgdG8gYSB3b3JrZXIgdGhyZWFkLlxuICAgICAqL1xuICAgIHByb3h5PzogYm9vbGVhbjtcbiAgfVxuXG4gIGV4cG9ydCBpbnRlcmZhY2UgV2ViR0xGbGFncyB7XG4gICAgLyoqXG4gICAgICogU2V0IG9yIGdldCB0aGUgV2ViR0wgQ29udGV4dCBJRCAod2ViZ2wgb3Igd2ViZ2wyKS4gRGVmYXVsdCB2YWx1ZSBpcyB3ZWJnbDIuXG4gICAgICovXG4gICAgY29udGV4dElkPzogJ3dlYmdsJ3wnd2ViZ2wyJztcbiAgICAvKipcbiAgICAgKiBTZXQgb3IgZ2V0IHRoZSBtYXhpbXVtIGJhdGNoIHNpemUgZm9yIG1hdG11bC4gMCBtZWFucyB0byBkaXNhYmxlIGJhdGNoaW5nLlxuICAgICAqL1xuICAgIG1hdG11bE1heEJhdGNoU2l6ZT86IG51bWJlcjtcbiAgICAvKipcbiAgICAgKiBTZXQgb3IgZ2V0IHRoZSB0ZXh0dXJlIGNhY2hlIG1vZGUuIERlZmF1bHQgdmFsdWUgaXMgZnVsbC5cbiAgICAgKi9cbiAgICB0ZXh0dXJlQ2FjaGVNb2RlPzogJ2luaXRpYWxpemVyT25seSd8J2Z1bGwnO1xuICAgIC8qKlxuICAgICAqIFNldCBvciBnZXQgdGhlIHBhY2tlZCB0ZXh0dXJlIG1vZGVcbiAgICAgKi9cbiAgICBwYWNrPzogYm9vbGVhbjtcbiAgICAvKipcbiAgICAgKiBTZXQgb3IgZ2V0IHdoZXRoZXIgZW5hYmxlIGFzeW5jIGRvd25sb2FkLlxuICAgICAqL1xuICAgIGFzeW5jPzogYm9vbGVhbjtcbiAgfVxufVxuXG5leHBvcnQgaW50ZXJmYWNlIEVudiB7XG4gIC8qKlxuICAgKiBzZXQgdGhlIHNldmVyaXR5IGxldmVsIGZvciBsb2dnaW5nLiBJZiBvbWl0dGVkLCBkZWZhdWx0IGlzICd3YXJuaW5nJ1xuICAgKi9cbiAgbG9nTGV2ZWw/OiAndmVyYm9zZSd8J2luZm8nfCd3YXJuaW5nJ3wnZXJyb3InfCdmYXRhbCc7XG4gIC8qKlxuICAgKiBJbmRpY2F0ZSB3aGV0aGVyIHJ1biBpbiBkZWJ1ZyBtb2RlLlxuICAgKi9cbiAgZGVidWc/OiBib29sZWFuO1xuXG4gIC8qKlxuICAgKiBSZXByZXNlbnQgYSBzZXQgb2YgZmxhZ3MgZm9yIFdlYkFzc2VtYmx5XG4gICAqL1xuICB3YXNtOiBFbnYuV2ViQXNzZW1ibHlGbGFncztcblxuICAvKipcbiAgICogUmVwcmVzZW50IGEgc2V0IG9mIGZsYWdzIGZvciBXZWJHTFxuICAgKi9cbiAgd2ViZ2w6IEVudi5XZWJHTEZsYWdzO1xuXG4gIFtuYW1lOiBzdHJpbmddOiB1bmtub3duO1xufVxuXG4vKipcbiAqIFJlcHJlc2VudCBhIHNldCBvZiBmbGFncyBhcyBhIGdsb2JhbCBzaW5nbGV0b24uXG4gKi9cbmV4cG9ydCBjb25zdCBlbnY6IEVudiA9IG5ldyBFbnZJbXBsKCk7XG4iLCIvLyBDb3B5cmlnaHQgKGMpIE1pY3Jvc29mdCBDb3Jwb3JhdGlvbi4gQWxsIHJpZ2h0cyByZXNlcnZlZC5cbi8vIExpY2Vuc2VkIHVuZGVyIHRoZSBNSVQgTGljZW5zZS5cblxuaW1wb3J0IHtTZXNzaW9uSGFuZGxlcn0gZnJvbSAnLi9iYWNrZW5kJztcbmltcG9ydCB7cmVzb2x2ZUJhY2tlbmR9IGZyb20gJy4vYmFja2VuZC1pbXBsJztcbmltcG9ydCB7SW5mZXJlbmNlU2Vzc2lvbiBhcyBJbmZlcmVuY2VTZXNzaW9uSW50ZXJmYWNlfSBmcm9tICcuL2luZmVyZW5jZS1zZXNzaW9uJztcbmltcG9ydCB7T25ueFZhbHVlfSBmcm9tICcuL29ubngtdmFsdWUnO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4vdGVuc29yJztcblxudHlwZSBTZXNzaW9uT3B0aW9ucyA9IEluZmVyZW5jZVNlc3Npb25JbnRlcmZhY2UuU2Vzc2lvbk9wdGlvbnM7XG50eXBlIFJ1bk9wdGlvbnMgPSBJbmZlcmVuY2VTZXNzaW9uSW50ZXJmYWNlLlJ1bk9wdGlvbnM7XG50eXBlIEZlZWRzVHlwZSA9IEluZmVyZW5jZVNlc3Npb25JbnRlcmZhY2UuRmVlZHNUeXBlO1xudHlwZSBGZXRjaGVzVHlwZSA9IEluZmVyZW5jZVNlc3Npb25JbnRlcmZhY2UuRmV0Y2hlc1R5cGU7XG50eXBlIFJldHVyblR5cGUgPSBJbmZlcmVuY2VTZXNzaW9uSW50ZXJmYWNlLlJldHVyblR5cGU7XG5cbmV4cG9ydCBjbGFzcyBJbmZlcmVuY2VTZXNzaW9uIGltcGxlbWVudHMgSW5mZXJlbmNlU2Vzc2lvbkludGVyZmFjZSB7XG4gIHByaXZhdGUgY29uc3RydWN0b3IoaGFuZGxlcjogU2Vzc2lvbkhhbmRsZXIpIHtcbiAgICB0aGlzLmhhbmRsZXIgPSBoYW5kbGVyO1xuICB9XG4gIHJ1bihmZWVkczogRmVlZHNUeXBlLCBvcHRpb25zPzogUnVuT3B0aW9ucyk6IFByb21pc2U8UmV0dXJuVHlwZT47XG4gIHJ1bihmZWVkczogRmVlZHNUeXBlLCBmZXRjaGVzOiBGZXRjaGVzVHlwZSwgb3B0aW9ucz86IFJ1bk9wdGlvbnMpOiBQcm9taXNlPFJldHVyblR5cGU+O1xuICBhc3luYyBydW4oZmVlZHM6IEZlZWRzVHlwZSwgYXJnMT86IEZldGNoZXNUeXBlfFJ1bk9wdGlvbnMsIGFyZzI/OiBSdW5PcHRpb25zKTogUHJvbWlzZTxSZXR1cm5UeXBlPiB7XG4gICAgY29uc3QgZmV0Y2hlczoge1tuYW1lOiBzdHJpbmddOiBPbm54VmFsdWV8bnVsbH0gPSB7fTtcbiAgICBsZXQgb3B0aW9uczogUnVuT3B0aW9ucyA9IHt9O1xuICAgIC8vIGNoZWNrIGlucHV0c1xuICAgIGlmICh0eXBlb2YgZmVlZHMgIT09ICdvYmplY3QnIHx8IGZlZWRzID09PSBudWxsIHx8IGZlZWRzIGluc3RhbmNlb2YgVGVuc29yIHx8IEFycmF5LmlzQXJyYXkoZmVlZHMpKSB7XG4gICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKFxuICAgICAgICAgICdcXCdmZWVkc1xcJyBtdXN0IGJlIGFuIG9iamVjdCB0aGF0IHVzZSBpbnB1dCBuYW1lcyBhcyBrZXlzIGFuZCBPbm54VmFsdWUgYXMgY29ycmVzcG9uZGluZyB2YWx1ZXMuJyk7XG4gICAgfVxuXG4gICAgbGV0IGlzRmV0Y2hlc0VtcHR5ID0gdHJ1ZTtcbiAgICAvLyBkZXRlcm1pbmUgd2hpY2ggb3ZlcnJpZGUgaXMgYmVpbmcgdXNlZFxuICAgIGlmICh0eXBlb2YgYXJnMSA9PT0gJ29iamVjdCcpIHtcbiAgICAgIGlmIChhcmcxID09PSBudWxsKSB7XG4gICAgICAgIHRocm93IG5ldyBUeXBlRXJyb3IoJ1VuZXhwZWN0ZWQgYXJndW1lbnRbMV06IGNhbm5vdCBiZSBudWxsLicpO1xuICAgICAgfVxuICAgICAgaWYgKGFyZzEgaW5zdGFuY2VvZiBUZW5zb3IpIHtcbiAgICAgICAgdGhyb3cgbmV3IFR5cGVFcnJvcignXFwnZmV0Y2hlc1xcJyBjYW5ub3QgYmUgYSBUZW5zb3InKTtcbiAgICAgIH1cblxuICAgICAgaWYgKEFycmF5LmlzQXJyYXkoYXJnMSkpIHtcbiAgICAgICAgaWYgKGFyZzEubGVuZ3RoID09PSAwKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFR5cGVFcnJvcignXFwnZmV0Y2hlc1xcJyBjYW5ub3QgYmUgYW4gZW1wdHkgYXJyYXkuJyk7XG4gICAgICAgIH1cbiAgICAgICAgaXNGZXRjaGVzRW1wdHkgPSBmYWxzZTtcbiAgICAgICAgLy8gb3V0cHV0IG5hbWVzXG4gICAgICAgIGZvciAoY29uc3QgbmFtZSBvZiBhcmcxKSB7XG4gICAgICAgICAgaWYgKHR5cGVvZiBuYW1lICE9PSAnc3RyaW5nJykge1xuICAgICAgICAgICAgdGhyb3cgbmV3IFR5cGVFcnJvcignXFwnZmV0Y2hlc1xcJyBtdXN0IGJlIGEgc3RyaW5nIGFycmF5IG9yIGFuIG9iamVjdC4nKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgaWYgKHRoaXMub3V0cHV0TmFtZXMuaW5kZXhPZihuYW1lKSA9PT0gLTEpIHtcbiAgICAgICAgICAgIHRocm93IG5ldyBSYW5nZUVycm9yKGAnZmV0Y2hlcycgY29udGFpbnMgaW52YWxpZCBvdXRwdXQgbmFtZTogJHtuYW1lfS5gKTtcbiAgICAgICAgICB9XG4gICAgICAgICAgZmV0Y2hlc1tuYW1lXSA9IG51bGw7XG4gICAgICAgIH1cblxuICAgICAgICBpZiAodHlwZW9mIGFyZzIgPT09ICdvYmplY3QnICYmIGFyZzIgIT09IG51bGwpIHtcbiAgICAgICAgICBvcHRpb25zID0gYXJnMjtcbiAgICAgICAgfSBlbHNlIGlmICh0eXBlb2YgYXJnMiAhPT0gJ3VuZGVmaW5lZCcpIHtcbiAgICAgICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKCdcXCdvcHRpb25zXFwnIG11c3QgYmUgYW4gb2JqZWN0LicpO1xuICAgICAgICB9XG4gICAgICB9IGVsc2Uge1xuICAgICAgICAvLyBkZWNpZGUgd2hldGhlciBhcmcxIGlzIGZldGNoZXMgb3Igb3B0aW9uc1xuICAgICAgICAvLyBpZiBhbnkgb3V0cHV0IG5hbWUgaXMgcHJlc2VudCBhbmQgaXRzIHZhbHVlIGlzIHZhbGlkIE9ubnhWYWx1ZSwgd2UgY29uc2lkZXIgaXQgZmV0Y2hlc1xuICAgICAgICBsZXQgaXNGZXRjaGVzID0gZmFsc2U7XG4gICAgICAgIGNvbnN0IGFyZzFLZXlzID0gT2JqZWN0LmdldE93blByb3BlcnR5TmFtZXMoYXJnMSk7XG4gICAgICAgIGZvciAoY29uc3QgbmFtZSBvZiB0aGlzLm91dHB1dE5hbWVzKSB7XG4gICAgICAgICAgaWYgKGFyZzFLZXlzLmluZGV4T2YobmFtZSkgIT09IC0xKSB7XG4gICAgICAgICAgICBjb25zdCB2ID0gKGFyZzEgYXMgSW5mZXJlbmNlU2Vzc2lvbkludGVyZmFjZS5OdWxsYWJsZU9ubnhWYWx1ZU1hcFR5cGUpW25hbWVdO1xuICAgICAgICAgICAgaWYgKHYgPT09IG51bGwgfHwgdiBpbnN0YW5jZW9mIFRlbnNvcikge1xuICAgICAgICAgICAgICBpc0ZldGNoZXMgPSB0cnVlO1xuICAgICAgICAgICAgICBpc0ZldGNoZXNFbXB0eSA9IGZhbHNlO1xuICAgICAgICAgICAgICBmZXRjaGVzW25hbWVdID0gdjtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cblxuICAgICAgICBpZiAoaXNGZXRjaGVzKSB7XG4gICAgICAgICAgaWYgKHR5cGVvZiBhcmcyID09PSAnb2JqZWN0JyAmJiBhcmcyICE9PSBudWxsKSB7XG4gICAgICAgICAgICBvcHRpb25zID0gYXJnMjtcbiAgICAgICAgICB9IGVsc2UgaWYgKHR5cGVvZiBhcmcyICE9PSAndW5kZWZpbmVkJykge1xuICAgICAgICAgICAgdGhyb3cgbmV3IFR5cGVFcnJvcignXFwnb3B0aW9uc1xcJyBtdXN0IGJlIGFuIG9iamVjdC4nKTtcbiAgICAgICAgICB9XG4gICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgb3B0aW9ucyA9IGFyZzEgYXMgUnVuT3B0aW9ucztcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH0gZWxzZSBpZiAodHlwZW9mIGFyZzEgIT09ICd1bmRlZmluZWQnKSB7XG4gICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKCdVbmV4cGVjdGVkIGFyZ3VtZW50WzFdOiBtdXN0IGJlIFxcJ2ZldGNoZXNcXCcgb3IgXFwnb3B0aW9uc1xcJy4nKTtcbiAgICB9XG5cbiAgICAvLyBjaGVjayBpZiBhbGwgaW5wdXRzIGFyZSBpbiBmZWVkXG4gICAgZm9yIChjb25zdCBuYW1lIG9mIHRoaXMuaW5wdXROYW1lcykge1xuICAgICAgaWYgKHR5cGVvZiBmZWVkc1tuYW1lXSA9PT0gJ3VuZGVmaW5lZCcpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKGBpbnB1dCAnJHtuYW1lfScgaXMgbWlzc2luZyBpbiAnZmVlZHMnLmApO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIGlmIG5vIGZldGNoZXMgaXMgc3BlY2lmaWVkLCB3ZSB1c2UgdGhlIGZ1bGwgb3V0cHV0IG5hbWVzIGxpc3RcbiAgICBpZiAoaXNGZXRjaGVzRW1wdHkpIHtcbiAgICAgIGZvciAoY29uc3QgbmFtZSBvZiB0aGlzLm91dHB1dE5hbWVzKSB7XG4gICAgICAgIGZldGNoZXNbbmFtZV0gPSBudWxsO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIGZlZWRzLCBmZXRjaGVzIGFuZCBvcHRpb25zIGFyZSBwcmVwYXJlZFxuXG4gICAgY29uc3QgcmVzdWx0cyA9IGF3YWl0IHRoaXMuaGFuZGxlci5ydW4oZmVlZHMsIGZldGNoZXMsIG9wdGlvbnMpO1xuICAgIGNvbnN0IHJldHVyblZhbHVlOiB7W25hbWU6IHN0cmluZ106IE9ubnhWYWx1ZX0gPSB7fTtcbiAgICBmb3IgKGNvbnN0IGtleSBpbiByZXN1bHRzKSB7XG4gICAgICBpZiAoT2JqZWN0Lmhhc093blByb3BlcnR5LmNhbGwocmVzdWx0cywga2V5KSkge1xuICAgICAgICByZXR1cm5WYWx1ZVtrZXldID0gbmV3IFRlbnNvcihyZXN1bHRzW2tleV0udHlwZSwgcmVzdWx0c1trZXldLmRhdGEsIHJlc3VsdHNba2V5XS5kaW1zKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIHJldHVyblZhbHVlO1xuICB9XG5cbiAgc3RhdGljIGNyZWF0ZShwYXRoOiBzdHJpbmcsIG9wdGlvbnM/OiBTZXNzaW9uT3B0aW9ucyk6IFByb21pc2U8SW5mZXJlbmNlU2Vzc2lvbkludGVyZmFjZT47XG4gIHN0YXRpYyBjcmVhdGUoYnVmZmVyOiBBcnJheUJ1ZmZlckxpa2UsIG9wdGlvbnM/OiBTZXNzaW9uT3B0aW9ucyk6IFByb21pc2U8SW5mZXJlbmNlU2Vzc2lvbkludGVyZmFjZT47XG4gIHN0YXRpYyBjcmVhdGUoYnVmZmVyOiBBcnJheUJ1ZmZlckxpa2UsIGJ5dGVPZmZzZXQ6IG51bWJlciwgYnl0ZUxlbmd0aD86IG51bWJlciwgb3B0aW9ucz86IFNlc3Npb25PcHRpb25zKTpcbiAgICAgIFByb21pc2U8SW5mZXJlbmNlU2Vzc2lvbkludGVyZmFjZT47XG4gIHN0YXRpYyBjcmVhdGUoYnVmZmVyOiBVaW50OEFycmF5LCBvcHRpb25zPzogU2Vzc2lvbk9wdGlvbnMpOiBQcm9taXNlPEluZmVyZW5jZVNlc3Npb25JbnRlcmZhY2U+O1xuICBzdGF0aWMgYXN5bmMgY3JlYXRlKFxuICAgICAgYXJnMDogc3RyaW5nfEFycmF5QnVmZmVyTGlrZXxVaW50OEFycmF5LCBhcmcxPzogU2Vzc2lvbk9wdGlvbnN8bnVtYmVyLCBhcmcyPzogbnVtYmVyLFxuICAgICAgYXJnMz86IFNlc3Npb25PcHRpb25zKTogUHJvbWlzZTxJbmZlcmVuY2VTZXNzaW9uSW50ZXJmYWNlPiB7XG4gICAgLy8gZWl0aGVyIGxvYWQgZnJvbSBhIGZpbGUgb3IgYnVmZmVyXG4gICAgbGV0IGZpbGVQYXRoT3JVaW50OEFycmF5OiBzdHJpbmd8VWludDhBcnJheTtcbiAgICBsZXQgb3B0aW9uczogU2Vzc2lvbk9wdGlvbnMgPSB7fTtcblxuICAgIGlmICh0eXBlb2YgYXJnMCA9PT0gJ3N0cmluZycpIHtcbiAgICAgIGZpbGVQYXRoT3JVaW50OEFycmF5ID0gYXJnMDtcbiAgICAgIGlmICh0eXBlb2YgYXJnMSA9PT0gJ29iamVjdCcgJiYgYXJnMSAhPT0gbnVsbCkge1xuICAgICAgICBvcHRpb25zID0gYXJnMTtcbiAgICAgIH0gZWxzZSBpZiAodHlwZW9mIGFyZzEgIT09ICd1bmRlZmluZWQnKSB7XG4gICAgICAgIHRocm93IG5ldyBUeXBlRXJyb3IoJ1xcJ29wdGlvbnNcXCcgbXVzdCBiZSBhbiBvYmplY3QuJyk7XG4gICAgICB9XG4gICAgfSBlbHNlIGlmIChhcmcwIGluc3RhbmNlb2YgVWludDhBcnJheSkge1xuICAgICAgZmlsZVBhdGhPclVpbnQ4QXJyYXkgPSBhcmcwO1xuICAgICAgaWYgKHR5cGVvZiBhcmcxID09PSAnb2JqZWN0JyAmJiBhcmcxICE9PSBudWxsKSB7XG4gICAgICAgIG9wdGlvbnMgPSBhcmcxO1xuICAgICAgfSBlbHNlIGlmICh0eXBlb2YgYXJnMSAhPT0gJ3VuZGVmaW5lZCcpIHtcbiAgICAgICAgdGhyb3cgbmV3IFR5cGVFcnJvcignXFwnb3B0aW9uc1xcJyBtdXN0IGJlIGFuIG9iamVjdC4nKTtcbiAgICAgIH1cbiAgICB9IGVsc2UgaWYgKFxuICAgICAgICBhcmcwIGluc3RhbmNlb2YgQXJyYXlCdWZmZXIgfHxcbiAgICAgICAgKHR5cGVvZiBTaGFyZWRBcnJheUJ1ZmZlciAhPT0gJ3VuZGVmaW5lZCcgJiYgYXJnMCBpbnN0YW5jZW9mIFNoYXJlZEFycmF5QnVmZmVyKSkge1xuICAgICAgY29uc3QgYnVmZmVyID0gYXJnMDtcbiAgICAgIGxldCBieXRlT2Zmc2V0ID0gMDtcbiAgICAgIGxldCBieXRlTGVuZ3RoID0gYXJnMC5ieXRlTGVuZ3RoO1xuICAgICAgaWYgKHR5cGVvZiBhcmcxID09PSAnb2JqZWN0JyAmJiBhcmcxICE9PSBudWxsKSB7XG4gICAgICAgIG9wdGlvbnMgPSBhcmcxO1xuICAgICAgfSBlbHNlIGlmICh0eXBlb2YgYXJnMSA9PT0gJ251bWJlcicpIHtcbiAgICAgICAgYnl0ZU9mZnNldCA9IGFyZzE7XG4gICAgICAgIGlmICghTnVtYmVyLmlzU2FmZUludGVnZXIoYnl0ZU9mZnNldCkpIHtcbiAgICAgICAgICB0aHJvdyBuZXcgUmFuZ2VFcnJvcignXFwnYnl0ZU9mZnNldFxcJyBtdXN0IGJlIGFuIGludGVnZXIuJyk7XG4gICAgICAgIH1cbiAgICAgICAgaWYgKGJ5dGVPZmZzZXQgPCAwIHx8IGJ5dGVPZmZzZXQgPj0gYnVmZmVyLmJ5dGVMZW5ndGgpIHtcbiAgICAgICAgICB0aHJvdyBuZXcgUmFuZ2VFcnJvcihgJ2J5dGVPZmZzZXQnIGlzIG91dCBvZiByYW5nZSBbMCwgJHtidWZmZXIuYnl0ZUxlbmd0aH0pLmApO1xuICAgICAgICB9XG4gICAgICAgIGJ5dGVMZW5ndGggPSBhcmcwLmJ5dGVMZW5ndGggLSBieXRlT2Zmc2V0O1xuICAgICAgICBpZiAodHlwZW9mIGFyZzIgPT09ICdudW1iZXInKSB7XG4gICAgICAgICAgYnl0ZUxlbmd0aCA9IGFyZzI7XG4gICAgICAgICAgaWYgKCFOdW1iZXIuaXNTYWZlSW50ZWdlcihieXRlTGVuZ3RoKSkge1xuICAgICAgICAgICAgdGhyb3cgbmV3IFJhbmdlRXJyb3IoJ1xcJ2J5dGVMZW5ndGhcXCcgbXVzdCBiZSBhbiBpbnRlZ2VyLicpO1xuICAgICAgICAgIH1cbiAgICAgICAgICBpZiAoYnl0ZUxlbmd0aCA8PSAwIHx8IGJ5dGVPZmZzZXQgKyBieXRlTGVuZ3RoID4gYnVmZmVyLmJ5dGVMZW5ndGgpIHtcbiAgICAgICAgICAgIHRocm93IG5ldyBSYW5nZUVycm9yKGAnYnl0ZUxlbmd0aCcgaXMgb3V0IG9mIHJhbmdlICgwLCAke2J1ZmZlci5ieXRlTGVuZ3RoIC0gYnl0ZU9mZnNldH1dLmApO1xuICAgICAgICAgIH1cbiAgICAgICAgICBpZiAodHlwZW9mIGFyZzMgPT09ICdvYmplY3QnICYmIGFyZzMgIT09IG51bGwpIHtcbiAgICAgICAgICAgIG9wdGlvbnMgPSBhcmczO1xuICAgICAgICAgIH0gZWxzZSBpZiAodHlwZW9mIGFyZzMgIT09ICd1bmRlZmluZWQnKSB7XG4gICAgICAgICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKCdcXCdvcHRpb25zXFwnIG11c3QgYmUgYW4gb2JqZWN0LicpO1xuICAgICAgICAgIH1cbiAgICAgICAgfSBlbHNlIGlmICh0eXBlb2YgYXJnMiAhPT0gJ3VuZGVmaW5lZCcpIHtcbiAgICAgICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKCdcXCdieXRlTGVuZ3RoXFwnIG11c3QgYmUgYSBudW1iZXIuJyk7XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSBpZiAodHlwZW9mIGFyZzEgIT09ICd1bmRlZmluZWQnKSB7XG4gICAgICAgIHRocm93IG5ldyBUeXBlRXJyb3IoJ1xcJ29wdGlvbnNcXCcgbXVzdCBiZSBhbiBvYmplY3QuJyk7XG4gICAgICB9XG4gICAgICBmaWxlUGF0aE9yVWludDhBcnJheSA9IG5ldyBVaW50OEFycmF5KGJ1ZmZlciwgYnl0ZU9mZnNldCwgYnl0ZUxlbmd0aCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRocm93IG5ldyBUeXBlRXJyb3IoJ1VuZXhwZWN0ZWQgYXJndW1lbnRbMF06IG11c3QgYmUgXFwncGF0aFxcJyBvciBcXCdidWZmZXJcXCcuJyk7XG4gICAgfVxuXG4gICAgLy8gZ2V0IGJhY2tlbmQgaGludHNcbiAgICBjb25zdCBlcHMgPSBvcHRpb25zLmV4ZWN1dGlvblByb3ZpZGVycyB8fCBbXTtcbiAgICBjb25zdCBiYWNrZW5kSGludHMgPSBlcHMubWFwKGkgPT4gdHlwZW9mIGkgPT09ICdzdHJpbmcnID8gaSA6IGkubmFtZSk7XG4gICAgY29uc3QgYmFja2VuZCA9IGF3YWl0IHJlc29sdmVCYWNrZW5kKGJhY2tlbmRIaW50cyk7XG4gICAgY29uc3QgaGFuZGxlciA9IGF3YWl0IGJhY2tlbmQuY3JlYXRlU2Vzc2lvbkhhbmRsZXIoZmlsZVBhdGhPclVpbnQ4QXJyYXksIG9wdGlvbnMpO1xuICAgIHJldHVybiBuZXcgSW5mZXJlbmNlU2Vzc2lvbihoYW5kbGVyKTtcbiAgfVxuXG4gIHN0YXJ0UHJvZmlsaW5nKCk6IHZvaWQge1xuICAgIHRoaXMuaGFuZGxlci5zdGFydFByb2ZpbGluZygpO1xuICB9XG4gIGVuZFByb2ZpbGluZygpOiB2b2lkIHtcbiAgICB0aGlzLmhhbmRsZXIuZW5kUHJvZmlsaW5nKCk7XG4gIH1cblxuICBnZXQgaW5wdXROYW1lcygpOiByZWFkb25seSBzdHJpbmdbXSB7XG4gICAgcmV0dXJuIHRoaXMuaGFuZGxlci5pbnB1dE5hbWVzO1xuICB9XG4gIGdldCBvdXRwdXROYW1lcygpOiByZWFkb25seSBzdHJpbmdbXSB7XG4gICAgcmV0dXJuIHRoaXMuaGFuZGxlci5vdXRwdXROYW1lcztcbiAgfVxuXG4gIHByaXZhdGUgaGFuZGxlcjogU2Vzc2lvbkhhbmRsZXI7XG59XG4iLCIvLyBDb3B5cmlnaHQgKGMpIE1pY3Jvc29mdCBDb3Jwb3JhdGlvbi4gQWxsIHJpZ2h0cyByZXNlcnZlZC5cbi8vIExpY2Vuc2VkIHVuZGVyIHRoZSBNSVQgTGljZW5zZS5cblxuaW1wb3J0IHtJbmZlcmVuY2VTZXNzaW9uIGFzIEluZmVyZW5jZVNlc3Npb25JbXBsfSBmcm9tICcuL2luZmVyZW5jZS1zZXNzaW9uLWltcGwnO1xuaW1wb3J0IHtPbm54VmFsdWV9IGZyb20gJy4vb25ueC12YWx1ZSc7XG5cbi8qIGVzbGludC1kaXNhYmxlIEB0eXBlc2NyaXB0LWVzbGludC9uby1yZWRlY2xhcmUgKi9cblxuZXhwb3J0IGRlY2xhcmUgbmFtZXNwYWNlIEluZmVyZW5jZVNlc3Npb24ge1xuICAvLyNyZWdpb24gaW5wdXQvb3V0cHV0IHR5cGVzXG5cbiAgdHlwZSBPbm54VmFsdWVNYXBUeXBlID0ge3JlYWRvbmx5IFtuYW1lOiBzdHJpbmddOiBPbm54VmFsdWV9O1xuICB0eXBlIE51bGxhYmxlT25ueFZhbHVlTWFwVHlwZSA9IHtyZWFkb25seSBbbmFtZTogc3RyaW5nXTogT25ueFZhbHVlIHwgbnVsbH07XG5cbiAgLyoqXG4gICAqIEEgZmVlZHMgKG1vZGVsIGlucHV0cykgaXMgYW4gb2JqZWN0IHRoYXQgdXNlcyBpbnB1dCBuYW1lcyBhcyBrZXlzIGFuZCBPbm54VmFsdWUgYXMgY29ycmVzcG9uZGluZyB2YWx1ZXMuXG4gICAqL1xuICB0eXBlIEZlZWRzVHlwZSA9IE9ubnhWYWx1ZU1hcFR5cGU7XG5cbiAgLyoqXG4gICAqIEEgZmV0Y2hlcyAobW9kZWwgb3V0cHV0cykgY291bGQgYmUgb25lIG9mIHRoZSBmb2xsb3dpbmc6XG4gICAqXG4gICAqIC0gT21pdHRlZC4gVXNlIG1vZGVsJ3Mgb3V0cHV0IG5hbWVzIGRlZmluaXRpb24uXG4gICAqIC0gQW4gYXJyYXkgb2Ygc3RyaW5nIGluZGljYXRpbmcgdGhlIG91dHB1dCBuYW1lcy5cbiAgICogLSBBbiBvYmplY3QgdGhhdCB1c2Ugb3V0cHV0IG5hbWVzIGFzIGtleXMgYW5kIE9ubnhWYWx1ZSBvciBudWxsIGFzIGNvcnJlc3BvbmRpbmcgdmFsdWVzLlxuICAgKlxuICAgKiBSRU1BUks6IGRpZmZlcmVudCBmcm9tIGlucHV0IGFyZ3VtZW50LCBpbiBvdXRwdXQsIE9ubnhWYWx1ZSBpcyBvcHRpb25hbC4gSWYgYW4gT25ueFZhbHVlIGlzIHByZXNlbnQgaXQgd2lsbCBiZVxuICAgKiB1c2VkIGFzIGEgcHJlLWFsbG9jYXRlZCB2YWx1ZSBieSB0aGUgaW5mZXJlbmNlIGVuZ2luZTsgaWYgb21pdHRlZCwgaW5mZXJlbmNlIGVuZ2luZSB3aWxsIGFsbG9jYXRlIGJ1ZmZlclxuICAgKiBpbnRlcm5hbGx5LlxuICAgKi9cbiAgdHlwZSBGZXRjaGVzVHlwZSA9IHJlYWRvbmx5IHN0cmluZ1tdfE51bGxhYmxlT25ueFZhbHVlTWFwVHlwZTtcblxuICAvKipcbiAgICogQSBpbmZlcmVuY2luZyByZXR1cm4gdHlwZSBpcyBhbiBvYmplY3QgdGhhdCB1c2VzIG91dHB1dCBuYW1lcyBhcyBrZXlzIGFuZCBPbm54VmFsdWUgYXMgY29ycmVzcG9uZGluZyB2YWx1ZXMuXG4gICAqL1xuICB0eXBlIFJldHVyblR5cGUgPSBPbm54VmFsdWVNYXBUeXBlO1xuXG4gIC8vI2VuZHJlZ2lvblxuXG4gIC8vI3JlZ2lvbiBzZXNzaW9uIG9wdGlvbnNcblxuICAvKipcbiAgICogQSBzZXQgb2YgY29uZmlndXJhdGlvbnMgZm9yIHNlc3Npb24gYmVoYXZpb3IuXG4gICAqL1xuICBleHBvcnQgaW50ZXJmYWNlIFNlc3Npb25PcHRpb25zIHtcbiAgICAvKipcbiAgICAgKiBBbiBhcnJheSBvZiBleGVjdXRpb24gcHJvdmlkZXIgb3B0aW9ucy5cbiAgICAgKlxuICAgICAqIEFuIGV4ZWN1dGlvbiBwcm92aWRlciBvcHRpb24gY2FuIGJlIGEgc3RyaW5nIGluZGljYXRpbmcgdGhlIG5hbWUgb2YgdGhlIGV4ZWN1dGlvbiBwcm92aWRlcixcbiAgICAgKiBvciBhbiBvYmplY3Qgb2YgY29ycmVzcG9uZGluZyB0eXBlLlxuICAgICAqL1xuICAgIGV4ZWN1dGlvblByb3ZpZGVycz86IHJlYWRvbmx5IEV4ZWN1dGlvblByb3ZpZGVyQ29uZmlnW107XG5cbiAgICAvKipcbiAgICAgKiBUaGUgaW50cmEgT1AgdGhyZWFkcyBudW1iZXIuXG4gICAgICpcbiAgICAgKiBUaGlzIHNldHRpbmcgaXMgYXZhaWxhYmxlIG9ubHkgaW4gT05OWFJ1bnRpbWUgKE5vZGUuanMgYmluZGluZyBhbmQgcmVhY3QtbmF0aXZlKS5cbiAgICAgKi9cbiAgICBpbnRyYU9wTnVtVGhyZWFkcz86IG51bWJlcjtcblxuICAgIC8qKlxuICAgICAqIFRoZSBpbnRlciBPUCB0aHJlYWRzIG51bWJlci5cbiAgICAgKlxuICAgICAqIFRoaXMgc2V0dGluZyBpcyBhdmFpbGFibGUgb25seSBpbiBPTk5YUnVudGltZSAoTm9kZS5qcyBiaW5kaW5nIGFuZCByZWFjdC1uYXRpdmUpLlxuICAgICAqL1xuICAgIGludGVyT3BOdW1UaHJlYWRzPzogbnVtYmVyO1xuXG4gICAgLyoqXG4gICAgICogVGhlIG9wdGltaXphdGlvbiBsZXZlbC5cbiAgICAgKlxuICAgICAqIFRoaXMgc2V0dGluZyBpcyBhdmFpbGFibGUgb25seSBpbiBPTk5YUnVudGltZSAoTm9kZS5qcyBiaW5kaW5nIGFuZCByZWFjdC1uYXRpdmUpIG9yIFdlYkFzc2VtYmx5IGJhY2tlbmRcbiAgICAgKi9cbiAgICBncmFwaE9wdGltaXphdGlvbkxldmVsPzogJ2Rpc2FibGVkJ3wnYmFzaWMnfCdleHRlbmRlZCd8J2FsbCc7XG5cbiAgICAvKipcbiAgICAgKiBXaGV0aGVyIGVuYWJsZSBDUFUgbWVtb3J5IGFyZW5hLlxuICAgICAqXG4gICAgICogVGhpcyBzZXR0aW5nIGlzIGF2YWlsYWJsZSBvbmx5IGluIE9OTlhSdW50aW1lIChOb2RlLmpzIGJpbmRpbmcgYW5kIHJlYWN0LW5hdGl2ZSkgb3IgV2ViQXNzZW1ibHkgYmFja2VuZFxuICAgICAqL1xuICAgIGVuYWJsZUNwdU1lbUFyZW5hPzogYm9vbGVhbjtcblxuICAgIC8qKlxuICAgICAqIFdoZXRoZXIgZW5hYmxlIG1lbW9yeSBwYXR0ZXJuLlxuICAgICAqXG4gICAgICogVGhpcyBzZXR0aW5nIGlzIGF2YWlsYWJsZSBvbmx5IGluIE9OTlhSdW50aW1lIChOb2RlLmpzIGJpbmRpbmcgYW5kIHJlYWN0LW5hdGl2ZSkgb3IgV2ViQXNzZW1ibHkgYmFja2VuZFxuICAgICAqL1xuICAgIGVuYWJsZU1lbVBhdHRlcm4/OiBib29sZWFuO1xuXG4gICAgLyoqXG4gICAgICogRXhlY3V0aW9uIG1vZGUuXG4gICAgICpcbiAgICAgKiBUaGlzIHNldHRpbmcgaXMgYXZhaWxhYmxlIG9ubHkgaW4gT05OWFJ1bnRpbWUgKE5vZGUuanMgYmluZGluZyBhbmQgcmVhY3QtbmF0aXZlKSBvciBXZWJBc3NlbWJseSBiYWNrZW5kXG4gICAgICovXG4gICAgZXhlY3V0aW9uTW9kZT86ICdzZXF1ZW50aWFsJ3wncGFyYWxsZWwnO1xuXG4gICAgLyoqXG4gICAgICogV2V0aGVyIGVuYWJsZSBwcm9maWxpbmcuXG4gICAgICpcbiAgICAgKiBUaGlzIHNldHRpbmcgaXMgYSBwbGFjZWhvbGRlciBmb3IgYSBmdXR1cmUgdXNlLlxuICAgICAqL1xuICAgIGVuYWJsZVByb2ZpbGluZz86IGJvb2xlYW47XG5cbiAgICAvKipcbiAgICAgKiBGaWxlIHByZWZpeCBmb3IgcHJvZmlsaW5nLlxuICAgICAqXG4gICAgICogVGhpcyBzZXR0aW5nIGlzIGEgcGxhY2Vob2xkZXIgZm9yIGEgZnV0dXJlIHVzZS5cbiAgICAgKi9cbiAgICBwcm9maWxlRmlsZVByZWZpeD86IHN0cmluZztcblxuICAgIC8qKlxuICAgICAqIExvZyBJRC5cbiAgICAgKlxuICAgICAqIFRoaXMgc2V0dGluZyBpcyBhdmFpbGFibGUgb25seSBpbiBPTk5YUnVudGltZSAoTm9kZS5qcyBiaW5kaW5nIGFuZCByZWFjdC1uYXRpdmUpIG9yIFdlYkFzc2VtYmx5IGJhY2tlbmRcbiAgICAgKi9cbiAgICBsb2dJZD86IHN0cmluZztcblxuICAgIC8qKlxuICAgICAqIExvZyBzZXZlcml0eSBsZXZlbC4gU2VlXG4gICAgICogaHR0cHM6Ly9naXRodWIuY29tL21pY3Jvc29mdC9vbm54cnVudGltZS9ibG9iL21hc3Rlci9pbmNsdWRlL29ubnhydW50aW1lL2NvcmUvY29tbW9uL2xvZ2dpbmcvc2V2ZXJpdHkuaFxuICAgICAqXG4gICAgICogVGhpcyBzZXR0aW5nIGlzIGF2YWlsYWJsZSBvbmx5IGluIE9OTlhSdW50aW1lIChOb2RlLmpzIGJpbmRpbmcgYW5kIHJlYWN0LW5hdGl2ZSkgb3IgV2ViQXNzZW1ibHkgYmFja2VuZFxuICAgICAqL1xuICAgIGxvZ1NldmVyaXR5TGV2ZWw/OiAwfDF8MnwzfDQ7XG5cbiAgICAvKipcbiAgICAgKiBMb2cgdmVyYm9zaXR5IGxldmVsLlxuICAgICAqXG4gICAgICogVGhpcyBzZXR0aW5nIGlzIGF2YWlsYWJsZSBvbmx5IGluIFdlYkFzc2VtYmx5IGJhY2tlbmQuIFdpbGwgc3VwcG9ydCBOb2RlLmpzIGJpbmRpbmcgYW5kIHJlYWN0LW5hdGl2ZSBsYXRlclxuICAgICAqL1xuICAgIGxvZ1ZlcmJvc2l0eUxldmVsPzogbnVtYmVyO1xuXG4gICAgLyoqXG4gICAgICogU3RvcmUgY29uZmlndXJhdGlvbnMgZm9yIGEgc2Vzc2lvbi4gU2VlXG4gICAgICogaHR0cHM6Ly9naXRodWIuY29tL21pY3Jvc29mdC9vbm54cnVudGltZS9ibG9iL21hc3Rlci9pbmNsdWRlL29ubnhydW50aW1lL2NvcmUvc2Vzc2lvbi9cbiAgICAgKiBvbm54cnVudGltZV9zZXNzaW9uX29wdGlvbnNfY29uZmlnX2tleXMuaFxuICAgICAqXG4gICAgICogSW4gZXhhbXBsZSxcbiAgICAgKlxuICAgICAqIGBgYGpzXG4gICAgICogZXh0cmE6IHtcbiAgICAgKiAgIHNlc3Npb246IHtcbiAgICAgKiAgICAgc2V0X2Rlbm9ybWFsX2FzX3plcm86IFwiMVwiLFxuICAgICAqICAgICBkaXNhYmxlX3ByZXBhY2tpbmc6IFwiMVwiXG4gICAgICogICB9LFxuICAgICAqICAgb3B0aW1pemF0aW9uOiB7XG4gICAgICogICAgIGVuYWJsZV9nZWx1X2FwcHJveGltYXRpb246IFwiMVwiXG4gICAgICogICB9XG4gICAgICogfVxuICAgICAqIGBgYFxuICAgICAqXG4gICAgICogVGhpcyBzZXR0aW5nIGlzIGF2YWlsYWJsZSBvbmx5IGluIFdlYkFzc2VtYmx5IGJhY2tlbmQuIFdpbGwgc3VwcG9ydCBOb2RlLmpzIGJpbmRpbmcgYW5kIHJlYWN0LW5hdGl2ZSBsYXRlclxuICAgICAqL1xuICAgIGV4dHJhPzogUmVjb3JkPHN0cmluZywgdW5rbm93bj47XG4gIH1cblxuICAvLyNyZWdpb24gZXhlY3V0aW9uIHByb3ZpZGVyc1xuXG4gIC8vIEN1cnJlbnRseSwgd2UgaGF2ZSB0aGUgZm9sbG93aW5nIGJhY2tlbmRzIHRvIHN1cHBvcnQgZXhlY3V0aW9uIHByb3ZpZGVyczpcbiAgLy8gQmFja2VuZCBOb2RlLmpzIGJpbmRpbmc6IHN1cHBvcnRzIFwiY3B1XCIgYW5kIFwiY3VkYVwiLlxuICAvLyBCYWNrZW5kIFdlYkFzc2VtYmx5OiBzdXBwb3J0cyAnd2FzbScuXG4gIC8vIEJhY2tlbmQgT05OWC5qczogc3VwcG9ydHMgJ3dlYmdsJy5cbiAgaW50ZXJmYWNlIEV4ZWN1dGlvblByb3ZpZGVyT3B0aW9uTWFwIHtcbiAgICBjcHU6IENwdUV4ZWN1dGlvblByb3ZpZGVyT3B0aW9uO1xuICAgIGN1ZGE6IEN1ZGFFeGVjdXRpb25Qcm92aWRlck9wdGlvbjtcbiAgICB3YXNtOiBXZWJBc3NlbWJseUV4ZWN1dGlvblByb3ZpZGVyT3B0aW9uO1xuICAgIHdlYmdsOiBXZWJHTEV4ZWN1dGlvblByb3ZpZGVyT3B0aW9uO1xuICAgIHdlYm5uOiBXZWJOTkV4ZWN1dGlvblByb3ZpZGVyT3B0aW9uO1xuICB9XG5cbiAgdHlwZSBFeGVjdXRpb25Qcm92aWRlck5hbWUgPSBrZXlvZiBFeGVjdXRpb25Qcm92aWRlck9wdGlvbk1hcDtcbiAgdHlwZSBFeGVjdXRpb25Qcm92aWRlckNvbmZpZyA9XG4gICAgICBFeGVjdXRpb25Qcm92aWRlck9wdGlvbk1hcFtFeGVjdXRpb25Qcm92aWRlck5hbWVdfEV4ZWN1dGlvblByb3ZpZGVyT3B0aW9ufEV4ZWN1dGlvblByb3ZpZGVyTmFtZXxzdHJpbmc7XG5cbiAgZXhwb3J0IGludGVyZmFjZSBFeGVjdXRpb25Qcm92aWRlck9wdGlvbiB7XG4gICAgcmVhZG9ubHkgbmFtZTogc3RyaW5nO1xuICB9XG4gIGV4cG9ydCBpbnRlcmZhY2UgQ3B1RXhlY3V0aW9uUHJvdmlkZXJPcHRpb24gZXh0ZW5kcyBFeGVjdXRpb25Qcm92aWRlck9wdGlvbiB7XG4gICAgcmVhZG9ubHkgbmFtZTogJ2NwdSc7XG4gICAgdXNlQXJlbmE/OiBib29sZWFuO1xuICB9XG4gIGV4cG9ydCBpbnRlcmZhY2UgQ3VkYUV4ZWN1dGlvblByb3ZpZGVyT3B0aW9uIGV4dGVuZHMgRXhlY3V0aW9uUHJvdmlkZXJPcHRpb24ge1xuICAgIHJlYWRvbmx5IG5hbWU6ICdjdWRhJztcbiAgICBkZXZpY2VJZD86IG51bWJlcjtcbiAgfVxuICBleHBvcnQgaW50ZXJmYWNlIFdlYkFzc2VtYmx5RXhlY3V0aW9uUHJvdmlkZXJPcHRpb24gZXh0ZW5kcyBFeGVjdXRpb25Qcm92aWRlck9wdGlvbiB7XG4gICAgcmVhZG9ubHkgbmFtZTogJ3dhc20nO1xuICAgIC8vIFRPRE86IGFkZCBmbGFnc1xuICB9XG4gIGV4cG9ydCBpbnRlcmZhY2UgV2ViR0xFeGVjdXRpb25Qcm92aWRlck9wdGlvbiBleHRlbmRzIEV4ZWN1dGlvblByb3ZpZGVyT3B0aW9uIHtcbiAgICByZWFkb25seSBuYW1lOiAnd2ViZ2wnO1xuICAgIC8vIFRPRE86IGFkZCBmbGFnc1xuICB9XG4gIGV4cG9ydCBpbnRlcmZhY2UgV2ViTk5FeGVjdXRpb25Qcm92aWRlck9wdGlvbiBleHRlbmRzIEV4ZWN1dGlvblByb3ZpZGVyT3B0aW9uIHtcbiAgICByZWFkb25seSBuYW1lOiAnd2Vibm4nO1xuICAgIGRldmljZVByZWZlcmVuY2U/OiBudW1iZXI7ICAvLyAwIC0gZGVmYXVsdCwgMSAtIGdwdSwgMiAtIGNwdVxuICB9XG4gIC8vI2VuZHJlZ2lvblxuXG4gIC8vI2VuZHJlZ2lvblxuXG4gIC8vI3JlZ2lvbiBydW4gb3B0aW9uc1xuXG4gIC8qKlxuICAgKiBBIHNldCBvZiBjb25maWd1cmF0aW9ucyBmb3IgaW5mZXJlbmNlIHJ1biBiZWhhdmlvclxuICAgKi9cbiAgZXhwb3J0IGludGVyZmFjZSBSdW5PcHRpb25zIHtcbiAgICAvKipcbiAgICAgKiBMb2cgc2V2ZXJpdHkgbGV2ZWwuIFNlZVxuICAgICAqIGh0dHBzOi8vZ2l0aHViLmNvbS9taWNyb3NvZnQvb25ueHJ1bnRpbWUvYmxvYi9tYXN0ZXIvaW5jbHVkZS9vbm54cnVudGltZS9jb3JlL2NvbW1vbi9sb2dnaW5nL3NldmVyaXR5LmhcbiAgICAgKlxuICAgICAqIFRoaXMgc2V0dGluZyBpcyBhdmFpbGFibGUgb25seSBpbiBPTk5YUnVudGltZSAoTm9kZS5qcyBiaW5kaW5nIGFuZCByZWFjdC1uYXRpdmUpIG9yIFdlYkFzc2VtYmx5IGJhY2tlbmRcbiAgICAgKi9cbiAgICBsb2dTZXZlcml0eUxldmVsPzogMHwxfDJ8M3w0O1xuXG4gICAgLyoqXG4gICAgICogTG9nIHZlcmJvc2l0eSBsZXZlbC5cbiAgICAgKlxuICAgICAqIFRoaXMgc2V0dGluZyBpcyBhdmFpbGFibGUgb25seSBpbiBXZWJBc3NlbWJseSBiYWNrZW5kLiBXaWxsIHN1cHBvcnQgTm9kZS5qcyBiaW5kaW5nIGFuZCByZWFjdC1uYXRpdmUgbGF0ZXJcbiAgICAgKi9cbiAgICBsb2dWZXJib3NpdHlMZXZlbD86IG51bWJlcjtcblxuICAgIC8qKlxuICAgICAqIFRlcm1pbmF0ZSBhbGwgaW5jb21wbGV0ZSBPcnRSdW4gY2FsbHMgYXMgc29vbiBhcyBwb3NzaWJsZSBpZiB0cnVlXG4gICAgICpcbiAgICAgKiBUaGlzIHNldHRpbmcgaXMgYXZhaWxhYmxlIG9ubHkgaW4gV2ViQXNzZW1ibHkgYmFja2VuZC4gV2lsbCBzdXBwb3J0IE5vZGUuanMgYmluZGluZyBhbmQgcmVhY3QtbmF0aXZlIGxhdGVyXG4gICAgICovXG4gICAgdGVybWluYXRlPzogYm9vbGVhbjtcblxuICAgIC8qKlxuICAgICAqIEEgdGFnIGZvciB0aGUgUnVuKCkgY2FsbHMgdXNpbmcgdGhpc1xuICAgICAqXG4gICAgICogVGhpcyBzZXR0aW5nIGlzIGF2YWlsYWJsZSBvbmx5IGluIE9OTlhSdW50aW1lIChOb2RlLmpzIGJpbmRpbmcgYW5kIHJlYWN0LW5hdGl2ZSkgb3IgV2ViQXNzZW1ibHkgYmFja2VuZFxuICAgICAqL1xuICAgIHRhZz86IHN0cmluZztcblxuICAgIC8qKlxuICAgICAqIFNldCBhIHNpbmdsZSBydW4gY29uZmlndXJhdGlvbiBlbnRyeS4gU2VlXG4gICAgICogaHR0cHM6Ly9naXRodWIuY29tL21pY3Jvc29mdC9vbm54cnVudGltZS9ibG9iL21hc3Rlci9pbmNsdWRlL29ubnhydW50aW1lL2NvcmUvc2Vzc2lvbi9cbiAgICAgKiBvbm54cnVudGltZV9ydW5fb3B0aW9uc19jb25maWdfa2V5cy5oXG4gICAgICpcbiAgICAgKiBJbiBleGFtcGxlLFxuICAgICAqXG4gICAgICogYGBganNcbiAgICAgKiBleHRyYToge1xuICAgICAqICAgbWVtb3J5OiB7XG4gICAgICogICAgIGVuYWJsZV9tZW1vcnlfYXJlbmFfc2hyaW5rYWdlOiBcIjFcIixcbiAgICAgKiAgIH1cbiAgICAgKiB9XG4gICAgICogYGBgXG4gICAgICpcbiAgICAgKiBUaGlzIHNldHRpbmcgaXMgYXZhaWxhYmxlIG9ubHkgaW4gV2ViQXNzZW1ibHkgYmFja2VuZC4gV2lsbCBzdXBwb3J0IE5vZGUuanMgYmluZGluZyBhbmQgcmVhY3QtbmF0aXZlIGxhdGVyXG4gICAgICovXG4gICAgZXh0cmE/OiBSZWNvcmQ8c3RyaW5nLCB1bmtub3duPjtcbiAgfVxuXG4gIC8vI2VuZHJlZ2lvblxuXG4gIC8vI3JlZ2lvbiB2YWx1ZSBtZXRhZGF0YVxuXG4gIC8vIGVzbGludC1kaXNhYmxlLW5leHQtbGluZSBAdHlwZXNjcmlwdC1lc2xpbnQvbm8tZW1wdHktaW50ZXJmYWNlXG4gIGludGVyZmFjZSBWYWx1ZU1ldGFkYXRhIHtcbiAgICAvLyBUQkRcbiAgfVxuXG4gIC8vI2VuZHJlZ2lvblxufVxuXG4vKipcbiAqIFJlcHJlc2VudCBhIHJ1bnRpbWUgaW5zdGFuY2Ugb2YgYW4gT05OWCBtb2RlbC5cbiAqL1xuZXhwb3J0IGludGVyZmFjZSBJbmZlcmVuY2VTZXNzaW9uIHtcbiAgLy8jcmVnaW9uIHJ1bigpXG5cbiAgLyoqXG4gICAqIEV4ZWN1dGUgdGhlIG1vZGVsIGFzeW5jaHJvbm91c2x5IHdpdGggdGhlIGdpdmVuIGZlZWRzIGFuZCBvcHRpb25zLlxuICAgKlxuICAgKiBAcGFyYW0gZmVlZHMgLSBSZXByZXNlbnRhdGlvbiBvZiB0aGUgbW9kZWwgaW5wdXQuIFNlZSB0eXBlIGRlc2NyaXB0aW9uIG9mIGBJbmZlcmVuY2VTZXNzaW9uLklucHV0VHlwZWAgZm9yIGRldGFpbC5cbiAgICogQHBhcmFtIG9wdGlvbnMgLSBPcHRpb25hbC4gQSBzZXQgb2Ygb3B0aW9ucyB0aGF0IGNvbnRyb2xzIHRoZSBiZWhhdmlvciBvZiBtb2RlbCBpbmZlcmVuY2UuXG4gICAqIEByZXR1cm5zIEEgcHJvbWlzZSB0aGF0IHJlc29sdmVzIHRvIGEgbWFwLCB3aGljaCB1c2VzIG91dHB1dCBuYW1lcyBhcyBrZXlzIGFuZCBPbm54VmFsdWUgYXMgY29ycmVzcG9uZGluZyB2YWx1ZXMuXG4gICAqL1xuICBydW4oZmVlZHM6IEluZmVyZW5jZVNlc3Npb24uRmVlZHNUeXBlLCBvcHRpb25zPzogSW5mZXJlbmNlU2Vzc2lvbi5SdW5PcHRpb25zKTogUHJvbWlzZTxJbmZlcmVuY2VTZXNzaW9uLlJldHVyblR5cGU+O1xuXG4gIC8qKlxuICAgKiBFeGVjdXRlIHRoZSBtb2RlbCBhc3luY2hyb25vdXNseSB3aXRoIHRoZSBnaXZlbiBmZWVkcywgZmV0Y2hlcyBhbmQgb3B0aW9ucy5cbiAgICpcbiAgICogQHBhcmFtIGZlZWRzIC0gUmVwcmVzZW50YXRpb24gb2YgdGhlIG1vZGVsIGlucHV0LiBTZWUgdHlwZSBkZXNjcmlwdGlvbiBvZiBgSW5mZXJlbmNlU2Vzc2lvbi5JbnB1dFR5cGVgIGZvciBkZXRhaWwuXG4gICAqIEBwYXJhbSBmZXRjaGVzIC0gUmVwcmVzZW50YXRpb24gb2YgdGhlIG1vZGVsIG91dHB1dC4gU2VlIHR5cGUgZGVzY3JpcHRpb24gb2YgYEluZmVyZW5jZVNlc3Npb24uT3V0cHV0VHlwZWAgZm9yXG4gICAqIGRldGFpbC5cbiAgICogQHBhcmFtIG9wdGlvbnMgLSBPcHRpb25hbC4gQSBzZXQgb2Ygb3B0aW9ucyB0aGF0IGNvbnRyb2xzIHRoZSBiZWhhdmlvciBvZiBtb2RlbCBpbmZlcmVuY2UuXG4gICAqIEByZXR1cm5zIEEgcHJvbWlzZSB0aGF0IHJlc29sdmVzIHRvIGEgbWFwLCB3aGljaCB1c2VzIG91dHB1dCBuYW1lcyBhcyBrZXlzIGFuZCBPbm54VmFsdWUgYXMgY29ycmVzcG9uZGluZyB2YWx1ZXMuXG4gICAqL1xuICBydW4oZmVlZHM6IEluZmVyZW5jZVNlc3Npb24uRmVlZHNUeXBlLCBmZXRjaGVzOiBJbmZlcmVuY2VTZXNzaW9uLkZldGNoZXNUeXBlLFxuICAgICAgb3B0aW9ucz86IEluZmVyZW5jZVNlc3Npb24uUnVuT3B0aW9ucyk6IFByb21pc2U8SW5mZXJlbmNlU2Vzc2lvbi5SZXR1cm5UeXBlPjtcblxuICAvLyNlbmRyZWdpb25cblxuICAvLyNyZWdpb24gcHJvZmlsaW5nXG5cbiAgLyoqXG4gICAqIFN0YXJ0IHByb2ZpbGluZy5cbiAgICovXG4gIHN0YXJ0UHJvZmlsaW5nKCk6IHZvaWQ7XG5cbiAgLyoqXG4gICAqIEVuZCBwcm9maWxpbmcuXG4gICAqL1xuICBlbmRQcm9maWxpbmcoKTogdm9pZDtcblxuICAvLyNlbmRyZWdpb25cblxuICAvLyNyZWdpb24gbWV0YWRhdGFcblxuICAvKipcbiAgICogR2V0IGlucHV0IG5hbWVzIG9mIHRoZSBsb2FkZWQgbW9kZWwuXG4gICAqL1xuICByZWFkb25seSBpbnB1dE5hbWVzOiByZWFkb25seSBzdHJpbmdbXTtcblxuICAvKipcbiAgICogR2V0IG91dHB1dCBuYW1lcyBvZiB0aGUgbG9hZGVkIG1vZGVsLlxuICAgKi9cbiAgcmVhZG9ubHkgb3V0cHV0TmFtZXM6IHJlYWRvbmx5IHN0cmluZ1tdO1xuXG4gIC8vIC8qKlxuICAvLyAgKiBHZXQgaW5wdXQgbWV0YWRhdGEgb2YgdGhlIGxvYWRlZCBtb2RlbC5cbiAgLy8gICovXG4gIC8vIHJlYWRvbmx5IGlucHV0TWV0YWRhdGE6IFJlYWRvbmx5QXJyYXk8UmVhZG9ubHk8SW5mZXJlbmNlU2Vzc2lvbi5WYWx1ZU1ldGFkYXRhPj47XG5cbiAgLy8gLyoqXG4gIC8vICAqIEdldCBvdXRwdXQgbWV0YWRhdGEgb2YgdGhlIGxvYWRlZCBtb2RlbC5cbiAgLy8gICovXG4gIC8vIHJlYWRvbmx5IG91dHB1dE1ldGFkYXRhOiBSZWFkb25seUFycmF5PFJlYWRvbmx5PEluZmVyZW5jZVNlc3Npb24uVmFsdWVNZXRhZGF0YT4+O1xuXG4gIC8vI2VuZHJlZ2lvblxufVxuXG5leHBvcnQgaW50ZXJmYWNlIEluZmVyZW5jZVNlc3Npb25GYWN0b3J5IHtcbiAgLy8jcmVnaW9uIGNyZWF0ZSgpXG5cbiAgLyoqXG4gICAqIENyZWF0ZSBhIG5ldyBpbmZlcmVuY2Ugc2Vzc2lvbiBhbmQgbG9hZCBtb2RlbCBhc3luY2hyb25vdXNseSBmcm9tIGFuIE9OTlggbW9kZWwgZmlsZS5cbiAgICpcbiAgICogQHBhcmFtIHVyaSAtIFRoZSBVUkkgb3IgZmlsZSBwYXRoIG9mIHRoZSBtb2RlbCB0byBsb2FkLlxuICAgKiBAcGFyYW0gb3B0aW9ucyAtIHNwZWNpZnkgY29uZmlndXJhdGlvbiBmb3IgY3JlYXRpbmcgYSBuZXcgaW5mZXJlbmNlIHNlc3Npb24uXG4gICAqIEByZXR1cm5zIEEgcHJvbWlzZSB0aGF0IHJlc29sdmVzIHRvIGFuIEluZmVyZW5jZVNlc3Npb24gb2JqZWN0LlxuICAgKi9cbiAgY3JlYXRlKHVyaTogc3RyaW5nLCBvcHRpb25zPzogSW5mZXJlbmNlU2Vzc2lvbi5TZXNzaW9uT3B0aW9ucyk6IFByb21pc2U8SW5mZXJlbmNlU2Vzc2lvbj47XG5cbiAgLyoqXG4gICAqIENyZWF0ZSBhIG5ldyBpbmZlcmVuY2Ugc2Vzc2lvbiBhbmQgbG9hZCBtb2RlbCBhc3luY2hyb25vdXNseSBmcm9tIGFuIGFycmF5IGJ1ZmVyLlxuICAgKlxuICAgKiBAcGFyYW0gYnVmZmVyIC0gQW4gQXJyYXlCdWZmZXIgcmVwcmVzZW50YXRpb24gb2YgYW4gT05OWCBtb2RlbC5cbiAgICogQHBhcmFtIG9wdGlvbnMgLSBzcGVjaWZ5IGNvbmZpZ3VyYXRpb24gZm9yIGNyZWF0aW5nIGEgbmV3IGluZmVyZW5jZSBzZXNzaW9uLlxuICAgKiBAcmV0dXJucyBBIHByb21pc2UgdGhhdCByZXNvbHZlcyB0byBhbiBJbmZlcmVuY2VTZXNzaW9uIG9iamVjdC5cbiAgICovXG4gIGNyZWF0ZShidWZmZXI6IEFycmF5QnVmZmVyTGlrZSwgb3B0aW9ucz86IEluZmVyZW5jZVNlc3Npb24uU2Vzc2lvbk9wdGlvbnMpOiBQcm9taXNlPEluZmVyZW5jZVNlc3Npb24+O1xuXG4gIC8qKlxuICAgKiBDcmVhdGUgYSBuZXcgaW5mZXJlbmNlIHNlc3Npb24gYW5kIGxvYWQgbW9kZWwgYXN5bmNocm9ub3VzbHkgZnJvbSBzZWdtZW50IG9mIGFuIGFycmF5IGJ1ZmVyLlxuICAgKlxuICAgKiBAcGFyYW0gYnVmZmVyIC0gQW4gQXJyYXlCdWZmZXIgcmVwcmVzZW50YXRpb24gb2YgYW4gT05OWCBtb2RlbC5cbiAgICogQHBhcmFtIGJ5dGVPZmZzZXQgLSBUaGUgYmVnaW5uaW5nIG9mIHRoZSBzcGVjaWZpZWQgcG9ydGlvbiBvZiB0aGUgYXJyYXkgYnVmZmVyLlxuICAgKiBAcGFyYW0gYnl0ZUxlbmd0aCAtIFRoZSBsZW5ndGggaW4gYnl0ZXMgb2YgdGhlIGFycmF5IGJ1ZmZlci5cbiAgICogQHBhcmFtIG9wdGlvbnMgLSBzcGVjaWZ5IGNvbmZpZ3VyYXRpb24gZm9yIGNyZWF0aW5nIGEgbmV3IGluZmVyZW5jZSBzZXNzaW9uLlxuICAgKiBAcmV0dXJucyBBIHByb21pc2UgdGhhdCByZXNvbHZlcyB0byBhbiBJbmZlcmVuY2VTZXNzaW9uIG9iamVjdC5cbiAgICovXG4gIGNyZWF0ZShidWZmZXI6IEFycmF5QnVmZmVyTGlrZSwgYnl0ZU9mZnNldDogbnVtYmVyLCBieXRlTGVuZ3RoPzogbnVtYmVyLCBvcHRpb25zPzogSW5mZXJlbmNlU2Vzc2lvbi5TZXNzaW9uT3B0aW9ucyk6XG4gICAgICBQcm9taXNlPEluZmVyZW5jZVNlc3Npb24+O1xuXG4gIC8qKlxuICAgKiBDcmVhdGUgYSBuZXcgaW5mZXJlbmNlIHNlc3Npb24gYW5kIGxvYWQgbW9kZWwgYXN5bmNocm9ub3VzbHkgZnJvbSBhIFVpbnQ4QXJyYXkuXG4gICAqXG4gICAqIEBwYXJhbSBidWZmZXIgLSBBIFVpbnQ4QXJyYXkgcmVwcmVzZW50YXRpb24gb2YgYW4gT05OWCBtb2RlbC5cbiAgICogQHBhcmFtIG9wdGlvbnMgLSBzcGVjaWZ5IGNvbmZpZ3VyYXRpb24gZm9yIGNyZWF0aW5nIGEgbmV3IGluZmVyZW5jZSBzZXNzaW9uLlxuICAgKiBAcmV0dXJucyBBIHByb21pc2UgdGhhdCByZXNvbHZlcyB0byBhbiBJbmZlcmVuY2VTZXNzaW9uIG9iamVjdC5cbiAgICovXG4gIGNyZWF0ZShidWZmZXI6IFVpbnQ4QXJyYXksIG9wdGlvbnM/OiBJbmZlcmVuY2VTZXNzaW9uLlNlc3Npb25PcHRpb25zKTogUHJvbWlzZTxJbmZlcmVuY2VTZXNzaW9uPjtcblxuICAvLyNlbmRyZWdpb25cbn1cblxuLy8gZXNsaW50LWRpc2FibGUtbmV4dC1saW5lIEB0eXBlc2NyaXB0LWVzbGludC9uYW1pbmctY29udmVudGlvblxuZXhwb3J0IGNvbnN0IEluZmVyZW5jZVNlc3Npb246IEluZmVyZW5jZVNlc3Npb25GYWN0b3J5ID0gSW5mZXJlbmNlU2Vzc2lvbkltcGw7XG4iLCIvLyBDb3B5cmlnaHQgKGMpIE1pY3Jvc29mdCBDb3Jwb3JhdGlvbi4gQWxsIHJpZ2h0cyByZXNlcnZlZC5cbi8vIExpY2Vuc2VkIHVuZGVyIHRoZSBNSVQgTGljZW5zZS5cblxuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4vdGVuc29yJztcblxudHlwZSBOb25UZW5zb3JUeXBlID0gbmV2ZXI7XG5cbi8qKlxuICogVHlwZSBPbm54VmFsdWUgUmVwcmVzZW50cyBib3RoIHRlbnNvcnMgYW5kIG5vbi10ZW5zb3JzIHZhbHVlIGZvciBtb2RlbCdzIGlucHV0cy9vdXRwdXRzLlxuICpcbiAqIE5PVEU6IGN1cnJlbnRseSBub3Qgc3VwcG9ydCBub24tdGVuc29yXG4gKi9cbmV4cG9ydCB0eXBlIE9ubnhWYWx1ZSA9IFRlbnNvcnxOb25UZW5zb3JUeXBlO1xuIiwiLy8gQ29weXJpZ2h0IChjKSBNaWNyb3NvZnQgQ29ycG9yYXRpb24uIEFsbCByaWdodHMgcmVzZXJ2ZWQuXG4vLyBMaWNlbnNlZCB1bmRlciB0aGUgTUlUIExpY2Vuc2UuXG5cbmltcG9ydCB7VGVuc29yIGFzIFRlbnNvckludGVyZmFjZX0gZnJvbSAnLi90ZW5zb3InO1xuXG50eXBlIFRlbnNvclR5cGUgPSBUZW5zb3JJbnRlcmZhY2UuVHlwZTtcbnR5cGUgVGVuc29yRGF0YVR5cGUgPSBUZW5zb3JJbnRlcmZhY2UuRGF0YVR5cGU7XG5cbnR5cGUgU3VwcG9ydGVkVHlwZWRBcnJheUNvbnN0cnVjdG9ycyA9IEZsb2F0MzJBcnJheUNvbnN0cnVjdG9yfFVpbnQ4QXJyYXlDb25zdHJ1Y3RvcnxJbnQ4QXJyYXlDb25zdHJ1Y3RvcnxcbiAgICBVaW50MTZBcnJheUNvbnN0cnVjdG9yfEludDE2QXJyYXlDb25zdHJ1Y3RvcnxJbnQzMkFycmF5Q29uc3RydWN0b3J8QmlnSW50NjRBcnJheUNvbnN0cnVjdG9yfFVpbnQ4QXJyYXlDb25zdHJ1Y3RvcnxcbiAgICBGbG9hdDY0QXJyYXlDb25zdHJ1Y3RvcnxVaW50MzJBcnJheUNvbnN0cnVjdG9yfEJpZ1VpbnQ2NEFycmF5Q29uc3RydWN0b3I7XG50eXBlIFN1cHBvcnRlZFR5cGVkQXJyYXkgPSBJbnN0YW5jZVR5cGU8U3VwcG9ydGVkVHlwZWRBcnJheUNvbnN0cnVjdG9ycz47XG5cbmNvbnN0IGlzQmlnSW50NjRBcnJheUF2YWlsYWJsZSA9IHR5cGVvZiBCaWdJbnQ2NEFycmF5ICE9PSAndW5kZWZpbmVkJyAmJiB0eXBlb2YgQmlnSW50NjRBcnJheS5mcm9tID09PSAnZnVuY3Rpb24nO1xuY29uc3QgaXNCaWdVaW50NjRBcnJheUF2YWlsYWJsZSA9IHR5cGVvZiBCaWdVaW50NjRBcnJheSAhPT0gJ3VuZGVmaW5lZCcgJiYgdHlwZW9mIEJpZ1VpbnQ2NEFycmF5LmZyb20gPT09ICdmdW5jdGlvbic7XG5cbi8vIGEgcnVudGltZSBtYXAgdGhhdCBtYXBzIHR5cGUgc3RyaW5nIHRvIFR5cGVkQXJyYXkgY29uc3RydWN0b3IuIFNob3VsZCBtYXRjaCBUZW5zb3IuRGF0YVR5cGVNYXAuXG5jb25zdCBOVU1FUklDX1RFTlNPUl9UWVBFX1RPX1RZUEVEQVJSQVlfTUFQID0gbmV3IE1hcDxzdHJpbmcsIFN1cHBvcnRlZFR5cGVkQXJyYXlDb25zdHJ1Y3RvcnM+KFtcbiAgWydmbG9hdDMyJywgRmxvYXQzMkFycmF5XSxcbiAgWyd1aW50OCcsIFVpbnQ4QXJyYXldLFxuICBbJ2ludDgnLCBJbnQ4QXJyYXldLFxuICBbJ3VpbnQxNicsIFVpbnQxNkFycmF5XSxcbiAgWydpbnQxNicsIEludDE2QXJyYXldLFxuICBbJ2ludDMyJywgSW50MzJBcnJheV0sXG4gIFsnYm9vbCcsIFVpbnQ4QXJyYXldLFxuICBbJ2Zsb2F0NjQnLCBGbG9hdDY0QXJyYXldLFxuICBbJ3VpbnQzMicsIFVpbnQzMkFycmF5XSxcbl0pO1xuXG4vLyBhIHJ1bnRpbWUgbWFwIHRoYXQgbWFwcyB0eXBlIHN0cmluZyB0byBUeXBlZEFycmF5IGNvbnN0cnVjdG9yLiBTaG91bGQgbWF0Y2ggVGVuc29yLkRhdGFUeXBlTWFwLlxuY29uc3QgTlVNRVJJQ19URU5TT1JfVFlQRURBUlJBWV9UT19UWVBFX01BUCA9IG5ldyBNYXA8U3VwcG9ydGVkVHlwZWRBcnJheUNvbnN0cnVjdG9ycywgVGVuc29yVHlwZT4oW1xuICBbRmxvYXQzMkFycmF5LCAnZmxvYXQzMiddLFxuICBbVWludDhBcnJheSwgJ3VpbnQ4J10sXG4gIFtJbnQ4QXJyYXksICdpbnQ4J10sXG4gIFtVaW50MTZBcnJheSwgJ3VpbnQxNiddLFxuICBbSW50MTZBcnJheSwgJ2ludDE2J10sXG4gIFtJbnQzMkFycmF5LCAnaW50MzInXSxcbiAgW0Zsb2F0NjRBcnJheSwgJ2Zsb2F0NjQnXSxcbiAgW1VpbnQzMkFycmF5LCAndWludDMyJ10sXG5dKTtcblxuaWYgKGlzQmlnSW50NjRBcnJheUF2YWlsYWJsZSkge1xuICBOVU1FUklDX1RFTlNPUl9UWVBFX1RPX1RZUEVEQVJSQVlfTUFQLnNldCgnaW50NjQnLCBCaWdJbnQ2NEFycmF5KTtcbiAgTlVNRVJJQ19URU5TT1JfVFlQRURBUlJBWV9UT19UWVBFX01BUC5zZXQoQmlnSW50NjRBcnJheSwgJ2ludDY0Jyk7XG59XG5pZiAoaXNCaWdVaW50NjRBcnJheUF2YWlsYWJsZSkge1xuICBOVU1FUklDX1RFTlNPUl9UWVBFX1RPX1RZUEVEQVJSQVlfTUFQLnNldCgndWludDY0JywgQmlnVWludDY0QXJyYXkpO1xuICBOVU1FUklDX1RFTlNPUl9UWVBFREFSUkFZX1RPX1RZUEVfTUFQLnNldChCaWdVaW50NjRBcnJheSwgJ3VpbnQ2NCcpO1xufVxuXG4vKipcbiAqIGNhbGN1bGF0ZSBzaXplIGZyb20gZGltcy5cbiAqXG4gKiBAcGFyYW0gZGltcyB0aGUgZGltcyBhcnJheS4gTWF5IGJlIGFuIGlsbGVnYWwgaW5wdXQuXG4gKi9cbmNvbnN0IGNhbGN1bGF0ZVNpemUgPSAoZGltczogcmVhZG9ubHkgdW5rbm93bltdKTogbnVtYmVyID0+IHtcbiAgbGV0IHNpemUgPSAxO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IGRpbXMubGVuZ3RoOyBpKyspIHtcbiAgICBjb25zdCBkaW0gPSBkaW1zW2ldO1xuICAgIGlmICh0eXBlb2YgZGltICE9PSAnbnVtYmVyJyB8fCAhTnVtYmVyLmlzU2FmZUludGVnZXIoZGltKSkge1xuICAgICAgdGhyb3cgbmV3IFR5cGVFcnJvcihgZGltc1ske2l9XSBtdXN0IGJlIGFuIGludGVnZXIsIGdvdDogJHtkaW19YCk7XG4gICAgfVxuICAgIGlmIChkaW0gPCAwKSB7XG4gICAgICB0aHJvdyBuZXcgUmFuZ2VFcnJvcihgZGltc1ske2l9XSBtdXN0IGJlIGEgbm9uLW5lZ2F0aXZlIGludGVnZXIsIGdvdDogJHtkaW19YCk7XG4gICAgfVxuICAgIHNpemUgKj0gZGltO1xuICB9XG4gIHJldHVybiBzaXplO1xufTtcblxuZXhwb3J0IGNsYXNzIFRlbnNvciBpbXBsZW1lbnRzIFRlbnNvckludGVyZmFjZSB7XG4gIC8vI3JlZ2lvbiBjb25zdHJ1Y3RvcnNcbiAgY29uc3RydWN0b3IodHlwZTogVGVuc29yVHlwZSwgZGF0YTogVGVuc29yRGF0YVR5cGV8cmVhZG9ubHkgbnVtYmVyW118cmVhZG9ubHkgYm9vbGVhbltdLCBkaW1zPzogcmVhZG9ubHkgbnVtYmVyW10pO1xuICBjb25zdHJ1Y3RvcihkYXRhOiBUZW5zb3JEYXRhVHlwZXxyZWFkb25seSBib29sZWFuW10sIGRpbXM/OiByZWFkb25seSBudW1iZXJbXSk7XG4gIGNvbnN0cnVjdG9yKFxuICAgICAgYXJnMDogVGVuc29yVHlwZXxUZW5zb3JEYXRhVHlwZXxyZWFkb25seSBib29sZWFuW10sIGFyZzE/OiBUZW5zb3JEYXRhVHlwZXxyZWFkb25seSBudW1iZXJbXXxyZWFkb25seSBib29sZWFuW10sXG4gICAgICBhcmcyPzogcmVhZG9ubHkgbnVtYmVyW10pIHtcbiAgICBsZXQgdHlwZTogVGVuc29yVHlwZTtcbiAgICBsZXQgZGF0YTogVGVuc29yRGF0YVR5cGU7XG4gICAgbGV0IGRpbXM6IHR5cGVvZiBhcmcxfHR5cGVvZiBhcmcyO1xuICAgIC8vIGNoZWNrIHdoZXRoZXIgYXJnMCBpcyB0eXBlIG9yIGRhdGFcbiAgICBpZiAodHlwZW9mIGFyZzAgPT09ICdzdHJpbmcnKSB7XG4gICAgICAvL1xuICAgICAgLy8gT3ZlcnJpZGU6IGNvbnN0cnVjdG9yKHR5cGUsIGRhdGEsIC4uLilcbiAgICAgIC8vXG4gICAgICB0eXBlID0gYXJnMDtcbiAgICAgIGRpbXMgPSBhcmcyO1xuICAgICAgaWYgKGFyZzAgPT09ICdzdHJpbmcnKSB7XG4gICAgICAgIC8vIHN0cmluZyB0ZW5zb3JcbiAgICAgICAgaWYgKCFBcnJheS5pc0FycmF5KGFyZzEpKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IFR5cGVFcnJvcignQSBzdHJpbmcgdGVuc29yXFwncyBkYXRhIG11c3QgYmUgYSBzdHJpbmcgYXJyYXkuJyk7XG4gICAgICAgIH1cbiAgICAgICAgLy8gd2UgZG9uJ3QgY2hlY2sgd2hldGhlciBldmVyeSBlbGVtZW50IGluIHRoZSBhcnJheSBpcyBzdHJpbmc7IHRoaXMgaXMgdG9vIHNsb3cuIHdlIGFzc3VtZSBpdCdzIGNvcnJlY3QgYW5kXG4gICAgICAgIC8vIGVycm9yIHdpbGwgYmUgcG9wdWxhdGVkIGF0IGluZmVyZW5jZVxuICAgICAgICBkYXRhID0gYXJnMTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIC8vIG51bWVyaWMgdGVuc29yXG4gICAgICAgIGNvbnN0IHR5cGVkQXJyYXlDb25zdHJ1Y3RvciA9IE5VTUVSSUNfVEVOU09SX1RZUEVfVE9fVFlQRURBUlJBWV9NQVAuZ2V0KGFyZzApO1xuICAgICAgICBpZiAodHlwZWRBcnJheUNvbnN0cnVjdG9yID09PSB1bmRlZmluZWQpIHtcbiAgICAgICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKGBVbnN1cHBvcnRlZCB0ZW5zb3IgdHlwZTogJHthcmcwfS5gKTtcbiAgICAgICAgfVxuICAgICAgICBpZiAoQXJyYXkuaXNBcnJheShhcmcxKSkge1xuICAgICAgICAgIC8vIHVzZSAnYXMgYW55JyBoZXJlIGJlY2F1c2UgVHlwZVNjcmlwdCdzIGNoZWNrIG9uIHR5cGUgb2YgJ1N1cHBvcnRlZFR5cGVkQXJyYXlDb25zdHJ1Y3RvcnMuZnJvbSgpJyBwcm9kdWNlc1xuICAgICAgICAgIC8vIGluY29ycmVjdCByZXN1bHRzLlxuICAgICAgICAgIC8vICd0eXBlZEFycmF5Q29uc3RydWN0b3InIHNob3VsZCBiZSBvbmUgb2YgdGhlIHR5cGVkIGFycmF5IHByb3RvdHlwZSBvYmplY3RzLlxuICAgICAgICAgIC8vIGVzbGludC1kaXNhYmxlLW5leHQtbGluZSBAdHlwZXNjcmlwdC1lc2xpbnQvbm8tZXhwbGljaXQtYW55XG4gICAgICAgICAgZGF0YSA9ICh0eXBlZEFycmF5Q29uc3RydWN0b3IgYXMgYW55KS5mcm9tKGFyZzEpO1xuICAgICAgICB9IGVsc2UgaWYgKGFyZzEgaW5zdGFuY2VvZiB0eXBlZEFycmF5Q29uc3RydWN0b3IpIHtcbiAgICAgICAgICBkYXRhID0gYXJnMTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKGBBICR7dHlwZX0gdGVuc29yJ3MgZGF0YSBtdXN0IGJlIHR5cGUgb2YgJHt0eXBlZEFycmF5Q29uc3RydWN0b3J9YCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgLy9cbiAgICAgIC8vIE92ZXJyaWRlOiBjb25zdHJ1Y3RvcihkYXRhLCAuLi4pXG4gICAgICAvL1xuICAgICAgZGltcyA9IGFyZzE7XG4gICAgICBpZiAoQXJyYXkuaXNBcnJheShhcmcwKSkge1xuICAgICAgICAvLyBvbmx5IGJvb2xlYW5bXSBhbmQgc3RyaW5nW10gaXMgc3VwcG9ydGVkXG4gICAgICAgIGlmIChhcmcwLmxlbmd0aCA9PT0gMCkge1xuICAgICAgICAgIHRocm93IG5ldyBUeXBlRXJyb3IoJ1RlbnNvciB0eXBlIGNhbm5vdCBiZSBpbmZlcnJlZCBmcm9tIGFuIGVtcHR5IGFycmF5LicpO1xuICAgICAgICB9XG4gICAgICAgIGNvbnN0IGZpcnN0RWxlbWVudFR5cGUgPSB0eXBlb2YgYXJnMFswXTtcbiAgICAgICAgaWYgKGZpcnN0RWxlbWVudFR5cGUgPT09ICdzdHJpbmcnKSB7XG4gICAgICAgICAgdHlwZSA9ICdzdHJpbmcnO1xuICAgICAgICAgIGRhdGEgPSBhcmcwO1xuICAgICAgICB9IGVsc2UgaWYgKGZpcnN0RWxlbWVudFR5cGUgPT09ICdib29sZWFuJykge1xuICAgICAgICAgIHR5cGUgPSAnYm9vbCc7XG4gICAgICAgICAgLy8gJ2FyZzAnIGlzIG9mIHR5cGUgJ2Jvb2xlYW5bXScuIFVpbnQ4QXJyYXkuZnJvbShib29sZWFuW10pIGFjdHVhbGx5IHdvcmtzLCBidXQgdHlwZXNjcmlwdCB0aGlua3MgdGhpcyBpc1xuICAgICAgICAgIC8vIHdyb25nIHR5cGUuIFdlIHVzZSAnYXMgYW55JyB0byBtYWtlIGl0IGhhcHB5LlxuICAgICAgICAgIC8vIGVzbGludC1kaXNhYmxlLW5leHQtbGluZSBAdHlwZXNjcmlwdC1lc2xpbnQvbm8tZXhwbGljaXQtYW55XG4gICAgICAgICAgZGF0YSA9IFVpbnQ4QXJyYXkuZnJvbShhcmcwIGFzIGFueVtdKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKGBJbnZhbGlkIGVsZW1lbnQgdHlwZSBvZiBkYXRhIGFycmF5OiAke2ZpcnN0RWxlbWVudFR5cGV9LmApO1xuICAgICAgICB9XG4gICAgICB9IGVsc2Uge1xuICAgICAgICAvLyBnZXQgdGVuc29yIHR5cGUgZnJvbSBUeXBlZEFycmF5XG4gICAgICAgIGNvbnN0IG1hcHBlZFR5cGUgPVxuICAgICAgICAgICAgTlVNRVJJQ19URU5TT1JfVFlQRURBUlJBWV9UT19UWVBFX01BUC5nZXQoYXJnMC5jb25zdHJ1Y3RvciBhcyBTdXBwb3J0ZWRUeXBlZEFycmF5Q29uc3RydWN0b3JzKTtcbiAgICAgICAgaWYgKG1hcHBlZFR5cGUgPT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgIHRocm93IG5ldyBUeXBlRXJyb3IoYFVuc3VwcG9ydGVkIHR5cGUgZm9yIHRlbnNvciBkYXRhOiAke2FyZzAuY29uc3RydWN0b3J9LmApO1xuICAgICAgICB9XG4gICAgICAgIHR5cGUgPSBtYXBwZWRUeXBlO1xuICAgICAgICBkYXRhID0gYXJnMCBhcyBTdXBwb3J0ZWRUeXBlZEFycmF5O1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIHR5cGUgYW5kIGRhdGEgaXMgcHJvY2Vzc2VkLCBub3cgcHJvY2Vzc2luZyBkaW1zXG4gICAgaWYgKGRpbXMgPT09IHVuZGVmaW5lZCkge1xuICAgICAgLy8gYXNzdW1lIDEtRCB0ZW5zb3IgaWYgZGltcyBvbWl0dGVkXG4gICAgICBkaW1zID0gW2RhdGEubGVuZ3RoXTtcbiAgICB9IGVsc2UgaWYgKCFBcnJheS5pc0FycmF5KGRpbXMpKSB7XG4gICAgICB0aHJvdyBuZXcgVHlwZUVycm9yKCdBIHRlbnNvclxcJ3MgZGltcyBtdXN0IGJlIGEgbnVtYmVyIGFycmF5Jyk7XG4gICAgfVxuXG4gICAgLy8gcGVyZm9ybSBjaGVja1xuICAgIGNvbnN0IHNpemUgPSBjYWxjdWxhdGVTaXplKGRpbXMpO1xuICAgIGlmIChzaXplICE9PSBkYXRhLmxlbmd0aCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBUZW5zb3IncyBzaXplKCR7c2l6ZX0pIGRvZXMgbm90IG1hdGNoIGRhdGEgbGVuZ3RoKCR7ZGF0YS5sZW5ndGh9KS5gKTtcbiAgICB9XG5cbiAgICB0aGlzLmRpbXMgPSBkaW1zIGFzIHJlYWRvbmx5IG51bWJlcltdO1xuICAgIHRoaXMudHlwZSA9IHR5cGU7XG4gICAgdGhpcy5kYXRhID0gZGF0YTtcbiAgICB0aGlzLnNpemUgPSBzaXplO1xuICB9XG4gIC8vI2VuZHJlZ2lvblxuXG4gIC8vI3JlZ2lvbiBmaWVsZHNcbiAgcmVhZG9ubHkgZGltczogcmVhZG9ubHkgbnVtYmVyW107XG4gIHJlYWRvbmx5IHR5cGU6IFRlbnNvclR5cGU7XG4gIHJlYWRvbmx5IGRhdGE6IFRlbnNvckRhdGFUeXBlO1xuICByZWFkb25seSBzaXplOiBudW1iZXI7XG4gIC8vI2VuZHJlZ2lvblxuXG4gIC8vI3JlZ2lvbiB0ZW5zb3IgdXRpbGl0aWVzXG4gIHJlc2hhcGUoZGltczogcmVhZG9ubHkgbnVtYmVyW10pOiBUZW5zb3Ige1xuICAgIHJldHVybiBuZXcgVGVuc29yKHRoaXMudHlwZSwgdGhpcy5kYXRhLCBkaW1zKTtcbiAgfVxuICAvLyNlbmRyZWdpb25cbn1cbiIsIi8vIENvcHlyaWdodCAoYykgTWljcm9zb2Z0IENvcnBvcmF0aW9uLiBBbGwgcmlnaHRzIHJlc2VydmVkLlxuLy8gTGljZW5zZWQgdW5kZXIgdGhlIE1JVCBMaWNlbnNlLlxuXG5pbXBvcnQge1RlbnNvciBhcyBUZW5zb3JJbXBsfSBmcm9tICcuL3RlbnNvci1pbXBsJztcbmltcG9ydCB7VHlwZWRUZW5zb3JVdGlsc30gZnJvbSAnLi90ZW5zb3ItdXRpbHMnO1xuXG4vKiBlc2xpbnQtZGlzYWJsZSBAdHlwZXNjcmlwdC1lc2xpbnQvbm8tcmVkZWNsYXJlICovXG5cbi8qKlxuICogcmVwcmVzZW50IGEgYmFzaWMgdGVuc29yIHdpdGggc3BlY2lmaWVkIGRpbWVuc2lvbnMgYW5kIGRhdGEgdHlwZS5cbiAqL1xuaW50ZXJmYWNlIFR5cGVkVGVuc29yQmFzZTxUIGV4dGVuZHMgVGVuc29yLlR5cGU+IHtcbiAgLyoqXG4gICAqIEdldCB0aGUgZGltZW5zaW9ucyBvZiB0aGUgdGVuc29yLlxuICAgKi9cbiAgcmVhZG9ubHkgZGltczogcmVhZG9ubHkgbnVtYmVyW107XG4gIC8qKlxuICAgKiBHZXQgdGhlIGRhdGEgdHlwZSBvZiB0aGUgdGVuc29yLlxuICAgKi9cbiAgcmVhZG9ubHkgdHlwZTogVDtcbiAgLyoqXG4gICAqIEdldCB0aGUgYnVmZmVyIGRhdGEgb2YgdGhlIHRlbnNvci5cbiAgICovXG4gIHJlYWRvbmx5IGRhdGE6IFRlbnNvci5EYXRhVHlwZU1hcFtUXTtcbn1cblxuZXhwb3J0IGRlY2xhcmUgbmFtZXNwYWNlIFRlbnNvciB7XG4gIGludGVyZmFjZSBEYXRhVHlwZU1hcCB7XG4gICAgZmxvYXQzMjogRmxvYXQzMkFycmF5O1xuICAgIHVpbnQ4OiBVaW50OEFycmF5O1xuICAgIGludDg6IEludDhBcnJheTtcbiAgICB1aW50MTY6IFVpbnQxNkFycmF5O1xuICAgIGludDE2OiBJbnQxNkFycmF5O1xuICAgIGludDMyOiBJbnQzMkFycmF5O1xuICAgIGludDY0OiBCaWdJbnQ2NEFycmF5O1xuICAgIHN0cmluZzogc3RyaW5nW107XG4gICAgYm9vbDogVWludDhBcnJheTtcbiAgICBmbG9hdDE2OiBuZXZlcjsgIC8vIGhvbGQgb24gdXNpbmcgVWludDE2QXJyYXkgYmVmb3JlIHdlIGhhdmUgYSBjb25jcmV0ZSBzb2x1dGlvbiBmb3IgZmxvYXQgMTZcbiAgICBmbG9hdDY0OiBGbG9hdDY0QXJyYXk7XG4gICAgdWludDMyOiBVaW50MzJBcnJheTtcbiAgICB1aW50NjQ6IEJpZ1VpbnQ2NEFycmF5O1xuICAgIC8vIGNvbXBsZXg2NDogbmV2ZXI7XG4gICAgLy8gY29tcGxleDEyODogbmV2ZXI7XG4gICAgLy8gYmZsb2F0MTY6IG5ldmVyO1xuICB9XG5cbiAgaW50ZXJmYWNlIEVsZW1lbnRUeXBlTWFwIHtcbiAgICBmbG9hdDMyOiBudW1iZXI7XG4gICAgdWludDg6IG51bWJlcjtcbiAgICBpbnQ4OiBudW1iZXI7XG4gICAgdWludDE2OiBudW1iZXI7XG4gICAgaW50MTY6IG51bWJlcjtcbiAgICBpbnQzMjogbnVtYmVyO1xuICAgIGludDY0OiBiaWdpbnQ7XG4gICAgc3RyaW5nOiBzdHJpbmc7XG4gICAgYm9vbDogYm9vbGVhbjtcbiAgICBmbG9hdDE2OiBuZXZlcjsgIC8vIGhvbGQgb24gYmVmb3JlIHdlIGhhdmUgYSBjb25jcmV0IHNvbHV0aW9uIGZvciBmbG9hdCAxNlxuICAgIGZsb2F0NjQ6IG51bWJlcjtcbiAgICB1aW50MzI6IG51bWJlcjtcbiAgICB1aW50NjQ6IGJpZ2ludDtcbiAgICAvLyBjb21wbGV4NjQ6IG5ldmVyO1xuICAgIC8vIGNvbXBsZXgxMjg6IG5ldmVyO1xuICAgIC8vIGJmbG9hdDE2OiBuZXZlcjtcbiAgfVxuXG4gIHR5cGUgRGF0YVR5cGUgPSBEYXRhVHlwZU1hcFtUeXBlXTtcbiAgdHlwZSBFbGVtZW50VHlwZSA9IEVsZW1lbnRUeXBlTWFwW1R5cGVdO1xuXG4gIC8qKlxuICAgKiByZXByZXNlbnQgdGhlIGRhdGEgdHlwZSBvZiBhIHRlbnNvclxuICAgKi9cbiAgZXhwb3J0IHR5cGUgVHlwZSA9IGtleW9mIERhdGFUeXBlTWFwO1xufVxuXG4vKipcbiAqIFJlcHJlc2VudCBtdWx0aS1kaW1lbnNpb25hbCBhcnJheXMgdG8gZmVlZCB0byBvciBmZXRjaCBmcm9tIG1vZGVsIGluZmVyZW5jaW5nLlxuICovXG5leHBvcnQgaW50ZXJmYWNlIFR5cGVkVGVuc29yPFQgZXh0ZW5kcyBUZW5zb3IuVHlwZT4gZXh0ZW5kcyBUeXBlZFRlbnNvckJhc2U8VD4sIFR5cGVkVGVuc29yVXRpbHM8VD4ge31cbi8qKlxuICogUmVwcmVzZW50IG11bHRpLWRpbWVuc2lvbmFsIGFycmF5cyB0byBmZWVkIHRvIG9yIGZldGNoIGZyb20gbW9kZWwgaW5mZXJlbmNpbmcuXG4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgVGVuc29yIGV4dGVuZHMgVHlwZWRUZW5zb3JCYXNlPFRlbnNvci5UeXBlPiwgVHlwZWRUZW5zb3JVdGlsczxUZW5zb3IuVHlwZT4ge31cblxuZXhwb3J0IGludGVyZmFjZSBUZW5zb3JDb25zdHJ1Y3RvciB7XG4gIC8vI3JlZ2lvbiBzcGVjaWZ5IGVsZW1lbnQgdHlwZVxuICAvKipcbiAgICogQ29uc3RydWN0IGEgbmV3IHN0cmluZyB0ZW5zb3Igb2JqZWN0IGZyb20gdGhlIGdpdmVuIHR5cGUsIGRhdGEgYW5kIGRpbXMuXG4gICAqXG4gICAqIEBwYXJhbSB0eXBlIC0gU3BlY2lmeSB0aGUgZWxlbWVudCB0eXBlLlxuICAgKiBAcGFyYW0gZGF0YSAtIFNwZWNpZnkgdGhlIHRlbnNvciBkYXRhXG4gICAqIEBwYXJhbSBkaW1zIC0gU3BlY2lmeSB0aGUgZGltZW5zaW9uIG9mIHRoZSB0ZW5zb3IuIElmIG9taXR0ZWQsIGEgMS1EIHRlbnNvciBpcyBhc3N1bWVkLlxuICAgKi9cbiAgbmV3KHR5cGU6ICdzdHJpbmcnLCBkYXRhOiBUZW5zb3IuRGF0YVR5cGVNYXBbJ3N0cmluZyddfHJlYWRvbmx5IHN0cmluZ1tdLFxuICAgICAgZGltcz86IHJlYWRvbmx5IG51bWJlcltdKTogVHlwZWRUZW5zb3I8J3N0cmluZyc+O1xuXG4gIC8qKlxuICAgKiBDb25zdHJ1Y3QgYSBuZXcgYm9vbCB0ZW5zb3Igb2JqZWN0IGZyb20gdGhlIGdpdmVuIHR5cGUsIGRhdGEgYW5kIGRpbXMuXG4gICAqXG4gICAqIEBwYXJhbSB0eXBlIC0gU3BlY2lmeSB0aGUgZWxlbWVudCB0eXBlLlxuICAgKiBAcGFyYW0gZGF0YSAtIFNwZWNpZnkgdGhlIHRlbnNvciBkYXRhXG4gICAqIEBwYXJhbSBkaW1zIC0gU3BlY2lmeSB0aGUgZGltZW5zaW9uIG9mIHRoZSB0ZW5zb3IuIElmIG9taXR0ZWQsIGEgMS1EIHRlbnNvciBpcyBhc3N1bWVkLlxuICAgKi9cbiAgbmV3KHR5cGU6ICdib29sJywgZGF0YTogVGVuc29yLkRhdGFUeXBlTWFwWydib29sJ118cmVhZG9ubHkgYm9vbGVhbltdLCBkaW1zPzogcmVhZG9ubHkgbnVtYmVyW10pOiBUeXBlZFRlbnNvcjwnYm9vbCc+O1xuXG4gIC8qKlxuICAgKiBDb25zdHJ1Y3QgYSBuZXcgbnVtZXJpYyB0ZW5zb3Igb2JqZWN0IGZyb20gdGhlIGdpdmVuIHR5cGUsIGRhdGEgYW5kIGRpbXMuXG4gICAqXG4gICAqIEBwYXJhbSB0eXBlIC0gU3BlY2lmeSB0aGUgZWxlbWVudCB0eXBlLlxuICAgKiBAcGFyYW0gZGF0YSAtIFNwZWNpZnkgdGhlIHRlbnNvciBkYXRhXG4gICAqIEBwYXJhbSBkaW1zIC0gU3BlY2lmeSB0aGUgZGltZW5zaW9uIG9mIHRoZSB0ZW5zb3IuIElmIG9taXR0ZWQsIGEgMS1EIHRlbnNvciBpcyBhc3N1bWVkLlxuICAgKi9cbiAgbmV3PFQgZXh0ZW5kcyBFeGNsdWRlPFRlbnNvci5UeXBlLCAnc3RyaW5nJ3wnYm9vbCc+PihcbiAgICAgIHR5cGU6IFQsIGRhdGE6IFRlbnNvci5EYXRhVHlwZU1hcFtUXXxyZWFkb25seSBudW1iZXJbXSwgZGltcz86IHJlYWRvbmx5IG51bWJlcltdKTogVHlwZWRUZW5zb3I8VD47XG4gIC8vI2VuZHJlZ2lvblxuXG4gIC8vI3JlZ2lvbiBpbmZlciBlbGVtZW50IHR5cGVzXG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdCBhIG5ldyBmbG9hdDMyIHRlbnNvciBvYmplY3QgZnJvbSB0aGUgZ2l2ZW4gZGF0YSBhbmQgZGltcy5cbiAgICpcbiAgICogQHBhcmFtIGRhdGEgLSBTcGVjaWZ5IHRoZSB0ZW5zb3IgZGF0YVxuICAgKiBAcGFyYW0gZGltcyAtIFNwZWNpZnkgdGhlIGRpbWVuc2lvbiBvZiB0aGUgdGVuc29yLiBJZiBvbWl0dGVkLCBhIDEtRCB0ZW5zb3IgaXMgYXNzdW1lZC5cbiAgICovXG4gIG5ldyhkYXRhOiBGbG9hdDMyQXJyYXksIGRpbXM/OiByZWFkb25seSBudW1iZXJbXSk6IFR5cGVkVGVuc29yPCdmbG9hdDMyJz47XG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdCBhIG5ldyBpbnQ4IHRlbnNvciBvYmplY3QgZnJvbSB0aGUgZ2l2ZW4gZGF0YSBhbmQgZGltcy5cbiAgICpcbiAgICogQHBhcmFtIGRhdGEgLSBTcGVjaWZ5IHRoZSB0ZW5zb3IgZGF0YVxuICAgKiBAcGFyYW0gZGltcyAtIFNwZWNpZnkgdGhlIGRpbWVuc2lvbiBvZiB0aGUgdGVuc29yLiBJZiBvbWl0dGVkLCBhIDEtRCB0ZW5zb3IgaXMgYXNzdW1lZC5cbiAgICovXG4gIG5ldyhkYXRhOiBJbnQ4QXJyYXksIGRpbXM/OiByZWFkb25seSBudW1iZXJbXSk6IFR5cGVkVGVuc29yPCdpbnQ4Jz47XG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdCBhIG5ldyB1aW50OCB0ZW5zb3Igb2JqZWN0IGZyb20gdGhlIGdpdmVuIGRhdGEgYW5kIGRpbXMuXG4gICAqXG4gICAqIEBwYXJhbSBkYXRhIC0gU3BlY2lmeSB0aGUgdGVuc29yIGRhdGFcbiAgICogQHBhcmFtIGRpbXMgLSBTcGVjaWZ5IHRoZSBkaW1lbnNpb24gb2YgdGhlIHRlbnNvci4gSWYgb21pdHRlZCwgYSAxLUQgdGVuc29yIGlzIGFzc3VtZWQuXG4gICAqL1xuICBuZXcoZGF0YTogVWludDhBcnJheSwgZGltcz86IHJlYWRvbmx5IG51bWJlcltdKTogVHlwZWRUZW5zb3I8J3VpbnQ4Jz47XG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdCBhIG5ldyB1aW50MTYgdGVuc29yIG9iamVjdCBmcm9tIHRoZSBnaXZlbiBkYXRhIGFuZCBkaW1zLlxuICAgKlxuICAgKiBAcGFyYW0gZGF0YSAtIFNwZWNpZnkgdGhlIHRlbnNvciBkYXRhXG4gICAqIEBwYXJhbSBkaW1zIC0gU3BlY2lmeSB0aGUgZGltZW5zaW9uIG9mIHRoZSB0ZW5zb3IuIElmIG9taXR0ZWQsIGEgMS1EIHRlbnNvciBpcyBhc3N1bWVkLlxuICAgKi9cbiAgbmV3KGRhdGE6IFVpbnQxNkFycmF5LCBkaW1zPzogcmVhZG9ubHkgbnVtYmVyW10pOiBUeXBlZFRlbnNvcjwndWludDE2Jz47XG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdCBhIG5ldyBpbnQxNiB0ZW5zb3Igb2JqZWN0IGZyb20gdGhlIGdpdmVuIGRhdGEgYW5kIGRpbXMuXG4gICAqXG4gICAqIEBwYXJhbSBkYXRhIC0gU3BlY2lmeSB0aGUgdGVuc29yIGRhdGFcbiAgICogQHBhcmFtIGRpbXMgLSBTcGVjaWZ5IHRoZSBkaW1lbnNpb24gb2YgdGhlIHRlbnNvci4gSWYgb21pdHRlZCwgYSAxLUQgdGVuc29yIGlzIGFzc3VtZWQuXG4gICAqL1xuICBuZXcoZGF0YTogSW50MTZBcnJheSwgZGltcz86IHJlYWRvbmx5IG51bWJlcltdKTogVHlwZWRUZW5zb3I8J2ludDE2Jz47XG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdCBhIG5ldyBpbnQzMiB0ZW5zb3Igb2JqZWN0IGZyb20gdGhlIGdpdmVuIGRhdGEgYW5kIGRpbXMuXG4gICAqXG4gICAqIEBwYXJhbSBkYXRhIC0gU3BlY2lmeSB0aGUgdGVuc29yIGRhdGFcbiAgICogQHBhcmFtIGRpbXMgLSBTcGVjaWZ5IHRoZSBkaW1lbnNpb24gb2YgdGhlIHRlbnNvci4gSWYgb21pdHRlZCwgYSAxLUQgdGVuc29yIGlzIGFzc3VtZWQuXG4gICAqL1xuICBuZXcoZGF0YTogSW50MzJBcnJheSwgZGltcz86IHJlYWRvbmx5IG51bWJlcltdKTogVHlwZWRUZW5zb3I8J2ludDMyJz47XG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdCBhIG5ldyBpbnQ2NCB0ZW5zb3Igb2JqZWN0IGZyb20gdGhlIGdpdmVuIGRhdGEgYW5kIGRpbXMuXG4gICAqXG4gICAqIEBwYXJhbSBkYXRhIC0gU3BlY2lmeSB0aGUgdGVuc29yIGRhdGFcbiAgICogQHBhcmFtIGRpbXMgLSBTcGVjaWZ5IHRoZSBkaW1lbnNpb24gb2YgdGhlIHRlbnNvci4gSWYgb21pdHRlZCwgYSAxLUQgdGVuc29yIGlzIGFzc3VtZWQuXG4gICAqL1xuICBuZXcoZGF0YTogQmlnSW50NjRBcnJheSwgZGltcz86IHJlYWRvbmx5IG51bWJlcltdKTogVHlwZWRUZW5zb3I8J2ludDY0Jz47XG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdCBhIG5ldyBzdHJpbmcgdGVuc29yIG9iamVjdCBmcm9tIHRoZSBnaXZlbiBkYXRhIGFuZCBkaW1zLlxuICAgKlxuICAgKiBAcGFyYW0gZGF0YSAtIFNwZWNpZnkgdGhlIHRlbnNvciBkYXRhXG4gICAqIEBwYXJhbSBkaW1zIC0gU3BlY2lmeSB0aGUgZGltZW5zaW9uIG9mIHRoZSB0ZW5zb3IuIElmIG9taXR0ZWQsIGEgMS1EIHRlbnNvciBpcyBhc3N1bWVkLlxuICAgKi9cbiAgbmV3KGRhdGE6IHJlYWRvbmx5IHN0cmluZ1tdLCBkaW1zPzogcmVhZG9ubHkgbnVtYmVyW10pOiBUeXBlZFRlbnNvcjwnc3RyaW5nJz47XG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdCBhIG5ldyBib29sIHRlbnNvciBvYmplY3QgZnJvbSB0aGUgZ2l2ZW4gZGF0YSBhbmQgZGltcy5cbiAgICpcbiAgICogQHBhcmFtIGRhdGEgLSBTcGVjaWZ5IHRoZSB0ZW5zb3IgZGF0YVxuICAgKiBAcGFyYW0gZGltcyAtIFNwZWNpZnkgdGhlIGRpbWVuc2lvbiBvZiB0aGUgdGVuc29yLiBJZiBvbWl0dGVkLCBhIDEtRCB0ZW5zb3IgaXMgYXNzdW1lZC5cbiAgICovXG4gIG5ldyhkYXRhOiByZWFkb25seSBib29sZWFuW10sIGRpbXM/OiByZWFkb25seSBudW1iZXJbXSk6IFR5cGVkVGVuc29yPCdib29sJz47XG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdCBhIG5ldyBmbG9hdDY0IHRlbnNvciBvYmplY3QgZnJvbSB0aGUgZ2l2ZW4gZGF0YSBhbmQgZGltcy5cbiAgICpcbiAgICogQHBhcmFtIGRhdGEgLSBTcGVjaWZ5IHRoZSB0ZW5zb3IgZGF0YVxuICAgKiBAcGFyYW0gZGltcyAtIFNwZWNpZnkgdGhlIGRpbWVuc2lvbiBvZiB0aGUgdGVuc29yLiBJZiBvbWl0dGVkLCBhIDEtRCB0ZW5zb3IgaXMgYXNzdW1lZC5cbiAgICovXG4gIG5ldyhkYXRhOiBGbG9hdDY0QXJyYXksIGRpbXM/OiByZWFkb25seSBudW1iZXJbXSk6IFR5cGVkVGVuc29yPCdmbG9hdDY0Jz47XG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdCBhIG5ldyB1aW50MzIgdGVuc29yIG9iamVjdCBmcm9tIHRoZSBnaXZlbiBkYXRhIGFuZCBkaW1zLlxuICAgKlxuICAgKiBAcGFyYW0gZGF0YSAtIFNwZWNpZnkgdGhlIHRlbnNvciBkYXRhXG4gICAqIEBwYXJhbSBkaW1zIC0gU3BlY2lmeSB0aGUgZGltZW5zaW9uIG9mIHRoZSB0ZW5zb3IuIElmIG9taXR0ZWQsIGEgMS1EIHRlbnNvciBpcyBhc3N1bWVkLlxuICAgKi9cbiAgbmV3KGRhdGE6IFVpbnQzMkFycmF5LCBkaW1zPzogcmVhZG9ubHkgbnVtYmVyW10pOiBUeXBlZFRlbnNvcjwndWludDMyJz47XG5cbiAgLyoqXG4gICAqIENvbnN0cnVjdCBhIG5ldyB1aW50NjQgdGVuc29yIG9iamVjdCBmcm9tIHRoZSBnaXZlbiBkYXRhIGFuZCBkaW1zLlxuICAgKlxuICAgKiBAcGFyYW0gZGF0YSAtIFNwZWNpZnkgdGhlIHRlbnNvciBkYXRhXG4gICAqIEBwYXJhbSBkaW1zIC0gU3BlY2lmeSB0aGUgZGltZW5zaW9uIG9mIHRoZSB0ZW5zb3IuIElmIG9taXR0ZWQsIGEgMS1EIHRlbnNvciBpcyBhc3N1bWVkLlxuICAgKi9cbiAgbmV3KGRhdGE6IEJpZ1VpbnQ2NEFycmF5LCBkaW1zPzogcmVhZG9ubHkgbnVtYmVyW10pOiBUeXBlZFRlbnNvcjwndWludDY0Jz47XG5cbiAgLy8jZW5kcmVnaW9uXG5cbiAgLy8jcmVnaW9uIGZhbGwgYmFjayB0byBub24tZ2VuZXJpYyB0ZW5zb3IgdHlwZSBkZWNsYXJhdGlvblxuXG4gIC8qKlxuICAgKiBDb25zdHJ1Y3QgYSBuZXcgdGVuc29yIG9iamVjdCBmcm9tIHRoZSBnaXZlbiB0eXBlLCBkYXRhIGFuZCBkaW1zLlxuICAgKlxuICAgKiBAcGFyYW0gdHlwZSAtIFNwZWNpZnkgdGhlIGVsZW1lbnQgdHlwZS5cbiAgICogQHBhcmFtIGRhdGEgLSBTcGVjaWZ5IHRoZSB0ZW5zb3IgZGF0YVxuICAgKiBAcGFyYW0gZGltcyAtIFNwZWNpZnkgdGhlIGRpbWVuc2lvbiBvZiB0aGUgdGVuc29yLiBJZiBvbWl0dGVkLCBhIDEtRCB0ZW5zb3IgaXMgYXNzdW1lZC5cbiAgICovXG4gIG5ldyh0eXBlOiBUZW5zb3IuVHlwZSwgZGF0YTogVGVuc29yLkRhdGFUeXBlfHJlYWRvbmx5IG51bWJlcltdfHJlYWRvbmx5IGJvb2xlYW5bXSwgZGltcz86IHJlYWRvbmx5IG51bWJlcltdKTogVGVuc29yO1xuXG4gIC8qKlxuICAgKiBDb25zdHJ1Y3QgYSBuZXcgdGVuc29yIG9iamVjdCBmcm9tIHRoZSBnaXZlbiBkYXRhIGFuZCBkaW1zLlxuICAgKlxuICAgKiBAcGFyYW0gZGF0YSAtIFNwZWNpZnkgdGhlIHRlbnNvciBkYXRhXG4gICAqIEBwYXJhbSBkaW1zIC0gU3BlY2lmeSB0aGUgZGltZW5zaW9uIG9mIHRoZSB0ZW5zb3IuIElmIG9taXR0ZWQsIGEgMS1EIHRlbnNvciBpcyBhc3N1bWVkLlxuICAgKi9cbiAgbmV3KGRhdGE6IFRlbnNvci5EYXRhVHlwZSwgZGltcz86IHJlYWRvbmx5IG51bWJlcltdKTogVGVuc29yO1xuICAvLyNlbmRyZWdpb25cbn1cblxuLy8gZXNsaW50LWRpc2FibGUtbmV4dC1saW5lIEB0eXBlc2NyaXB0LWVzbGludC9uYW1pbmctY29udmVudGlvblxuZXhwb3J0IGNvbnN0IFRlbnNvciA9IFRlbnNvckltcGwgYXMgVGVuc29yQ29uc3RydWN0b3I7XG4iLCIvLyBUaGUgbW9kdWxlIGNhY2hlXG52YXIgX193ZWJwYWNrX21vZHVsZV9jYWNoZV9fID0ge307XG5cbi8vIFRoZSByZXF1aXJlIGZ1bmN0aW9uXG5mdW5jdGlvbiBfX3dlYnBhY2tfcmVxdWlyZV9fKG1vZHVsZUlkKSB7XG5cdC8vIENoZWNrIGlmIG1vZHVsZSBpcyBpbiBjYWNoZVxuXHR2YXIgY2FjaGVkTW9kdWxlID0gX193ZWJwYWNrX21vZHVsZV9jYWNoZV9fW21vZHVsZUlkXTtcblx0aWYgKGNhY2hlZE1vZHVsZSAhPT0gdW5kZWZpbmVkKSB7XG5cdFx0cmV0dXJuIGNhY2hlZE1vZHVsZS5leHBvcnRzO1xuXHR9XG5cdC8vIENyZWF0ZSBhIG5ldyBtb2R1bGUgKGFuZCBwdXQgaXQgaW50byB0aGUgY2FjaGUpXG5cdHZhciBtb2R1bGUgPSBfX3dlYnBhY2tfbW9kdWxlX2NhY2hlX19bbW9kdWxlSWRdID0ge1xuXHRcdC8vIG5vIG1vZHVsZS5pZCBuZWVkZWRcblx0XHQvLyBubyBtb2R1bGUubG9hZGVkIG5lZWRlZFxuXHRcdGV4cG9ydHM6IHt9XG5cdH07XG5cblx0Ly8gRXhlY3V0ZSB0aGUgbW9kdWxlIGZ1bmN0aW9uXG5cdF9fd2VicGFja19tb2R1bGVzX19bbW9kdWxlSWRdKG1vZHVsZSwgbW9kdWxlLmV4cG9ydHMsIF9fd2VicGFja19yZXF1aXJlX18pO1xuXG5cdC8vIFJldHVybiB0aGUgZXhwb3J0cyBvZiB0aGUgbW9kdWxlXG5cdHJldHVybiBtb2R1bGUuZXhwb3J0cztcbn1cblxuIiwiLy8gZGVmaW5lIGdldHRlciBmdW5jdGlvbnMgZm9yIGhhcm1vbnkgZXhwb3J0c1xuX193ZWJwYWNrX3JlcXVpcmVfXy5kID0gZnVuY3Rpb24oZXhwb3J0cywgZGVmaW5pdGlvbikge1xuXHRmb3IodmFyIGtleSBpbiBkZWZpbml0aW9uKSB7XG5cdFx0aWYoX193ZWJwYWNrX3JlcXVpcmVfXy5vKGRlZmluaXRpb24sIGtleSkgJiYgIV9fd2VicGFja19yZXF1aXJlX18ubyhleHBvcnRzLCBrZXkpKSB7XG5cdFx0XHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywga2V5LCB7IGVudW1lcmFibGU6IHRydWUsIGdldDogZGVmaW5pdGlvbltrZXldIH0pO1xuXHRcdH1cblx0fVxufTsiLCJfX3dlYnBhY2tfcmVxdWlyZV9fLm8gPSBmdW5jdGlvbihvYmosIHByb3ApIHsgcmV0dXJuIE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChvYmosIHByb3ApOyB9IiwiLy8gZGVmaW5lIF9fZXNNb2R1bGUgb24gZXhwb3J0c1xuX193ZWJwYWNrX3JlcXVpcmVfXy5yID0gZnVuY3Rpb24oZXhwb3J0cykge1xuXHRpZih0eXBlb2YgU3ltYm9sICE9PSAndW5kZWZpbmVkJyAmJiBTeW1ib2wudG9TdHJpbmdUYWcpIHtcblx0XHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgU3ltYm9sLnRvU3RyaW5nVGFnLCB7IHZhbHVlOiAnTW9kdWxlJyB9KTtcblx0fVxuXHRPYmplY3QuZGVmaW5lUHJvcGVydHkoZXhwb3J0cywgJ19fZXNNb2R1bGUnLCB7IHZhbHVlOiB0cnVlIH0pO1xufTsiLCIvLyBDb3B5cmlnaHQgKGMpIE1pY3Jvc29mdCBDb3Jwb3JhdGlvbi4gQWxsIHJpZ2h0cyByZXNlcnZlZC5cbi8vIExpY2Vuc2VkIHVuZGVyIHRoZSBNSVQgTGljZW5zZS5cblxuZXhwb3J0ICogZnJvbSAnLi9iYWNrZW5kJztcbmV4cG9ydCAqIGZyb20gJy4vZW52JztcbmV4cG9ydCAqIGZyb20gJy4vaW5mZXJlbmNlLXNlc3Npb24nO1xuZXhwb3J0ICogZnJvbSAnLi90ZW5zb3InO1xuZXhwb3J0ICogZnJvbSAnLi9vbm54LXZhbHVlJztcbiJdLCJzb3VyY2VSb290IjoiIn0=