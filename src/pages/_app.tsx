import Head from 'next/head';
import { AppProps } from 'next/app';

import './styles.css';
import styles from './MainLayout.module.css';

const title = 'WebGPU Crowd Simulation';

const MainLayout: React.FunctionComponent<AppProps> = ({
  Component,
  pageProps,
}) => {

  return (
    <>
      <Head>
        <title>{title}</title>
        <meta
          name="description"
          content="This WebGPU crowd simulation was developed by Ashley Alexander-Lee, Matt Elser, and Wayne Wu."
        />
        <meta
          name="viewport"
          content="width=device-width, initial-scale=1, shrink-to-fit=no"
        />
      </Head>
      <div className={styles.wrapper}>
        <Component {...pageProps} />
      </div>
    </>
  );
};

export default MainLayout;
