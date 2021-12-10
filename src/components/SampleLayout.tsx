import Head from 'next/head';
import { useRouter } from 'next/router';
import { useEffect, useMemo, useRef, useState } from 'react';

import type { GUI } from 'dat.gui';
import * as Stats from 'stats.js';
import type { Editor, EditorConfiguration } from 'codemirror';
interface CodeMirrorEditor extends Editor {
  updatedSource: (source: string) => void;
}

import styles from './SampleLayout.module.css';

type SourceFileInfo = {
  name: string;
  contents: string;
  editable?: boolean;
};

export type SampleInit = (params: {
  canvasRef: React.RefObject<HTMLCanvasElement>;
  gui?: GUI;
  stats?: Stats;
}) => void | Promise<void>;

const setShaderRegisteredCallback =
  process.browser &&
  typeof GPUDevice !== 'undefined' &&
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  require('webgpu-live-shader-module').setShaderRegisteredCallback;

if (process.browser) {
  require('codemirror/mode/javascript/javascript');
}

function makeCodeMirrorEditor(source: string, editable: boolean) {
  const configuration: EditorConfiguration = {
    lineNumbers: true,
    lineWrapping: true,
    theme: 'monokai',
    readOnly: !editable,
  };

  let el: HTMLDivElement | null = null;
  let editor: CodeMirrorEditor;

  const updateCallbacks: ((source: string) => void)[] = [];

  if (process.browser) {
    el = document.createElement('div');
    const CodeMirror = process.browser && require('codemirror');
    editor = CodeMirror(el, configuration);
    editor.updatedSource = function (source) {
      updateCallbacks.forEach((cb) => cb(source));
    };
  }

  function Container(props: React.ComponentProps<'div'>) {
    return (
      <div {...props}>
        {editable ? (
          <button
            className={styles.updateSourceBtn}
            onClick={() => {
              editor.updatedSource(editor.getValue());
            }}
          >
            Update
          </button>
        ) : null}
        <div
          ref={(div) => {
            if (el && div) {
              div.appendChild(el);
              editor.setOption('value', source);
            }
          }}
        />
      </div>
    );
  }
  return {
    updateCallbacks,
    Container,
  };
}

const SampleLayout: React.FunctionComponent<
  React.PropsWithChildren<{
    name: string;
    description: string;
    filename: string;
    gui?: boolean;
    stats?: boolean;
    init: SampleInit;
    sources: SourceFileInfo[];
  }>
> = (props) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sources = useMemo(
    () =>
      props.sources.map(({ name, contents, editable }) => {
        return { name, ...makeCodeMirrorEditor(contents, editable) };
      }),
    props.sources
  );

  const guiParentRef = useRef<HTMLDivElement | null>(null);
  const gui: GUI | undefined = useMemo(() => {
    if (props.gui && process.browser) {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const dat = require('dat.gui');
      return new dat.GUI({ autoPlace: false });
    }
    return undefined;
  }, []);

  const statsParentRef = useRef<HTMLDivElement | null>(null);
  const stats: Stats | undefined = useMemo(() => {
    if (props.stats && process.browser) {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const stats = require('stats.js');
      return new stats();
    }
    return undefined;
  }, []);

  const router = useRouter();
  const currentHash = router.asPath.match(/#([a-zA-Z0-9\.\/]+)/);

  const [error, setError] = useState<unknown | null>(null);

  const canvasWidth = canvasRef.current != null ? canvasRef.current.width : 1600;
  const canvasHeight = canvasRef.current != null ? canvasRef.current.height : 800;

  const [activeHash, setActiveHash] = useState<string | null>(null);
  useEffect(() => {
    if (currentHash) {
      setActiveHash(currentHash[1]);
    } else {
      setActiveHash(sources[0].name);
    }

    if (gui && guiParentRef.current) {
      guiParentRef.current.appendChild(gui.domElement);
    }

    if (stats && statsParentRef) {
      statsParentRef.current.appendChild(stats.domElement);
    }

    try {
      const p = props.init({
        canvasRef,
        gui,
        stats
      });

      if (p instanceof Promise) {
        p.catch((err: Error) => {
          console.error(err);
          setError(err);
        });
      }
    } catch (err) {
      console.error(err);
      setError(err);
    }
  }, []);

  useEffect(() => {
    if (setShaderRegisteredCallback) {
      setShaderRegisteredCallback((source: string, updatedSource) => {
        const index = props.sources.findIndex(
          ({ contents }) => contents == source
        );
        //sources[index].updateCallbacks.push(updatedSource);
      });
    }
  }, [sources]);

  return (
    <main>
      <Head>
        <style
          dangerouslySetInnerHTML={{
            __html: `
            .CodeMirror {
              height: auto !important;
              margin: 1em 0;
            }

            .CodeMirror-scroll {
              height: auto !important;
              overflow: visible !important;
            }
          `,
          }}
        />
        <title className={styles.title}>{`${props.name}`}</title>
        <meta name="description" content={props.description} />
      </Head>
      <div className={styles.canvasContainer}>
        <div
          style={{
            position: 'absolute',
            right: 0
          }}
          ref={guiParentRef}
        ></div>
        <div
          style={{
            position: 'absolute',
            left: 10,
          }}
          ref={statsParentRef}
        ></div>
        <canvas ref={canvasRef} width={canvasWidth} height={canvasHeight}></canvas>
      </div>
    </main>
  );
};

export default SampleLayout;

export const makeSample: (
  ...props: Parameters<typeof SampleLayout>
) => JSX.Element = (props) => {
  return <SampleLayout {...props} />;
};
