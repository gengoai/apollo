package com.gengoai.apollo.ml.sequence;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author David B. Bracewell
 */
public enum CrfSuiteLoader {
   INSTANCE;

   private AtomicBoolean loaded = new AtomicBoolean(false);

   public void load() {
      if (!loaded.get()) {
         synchronized (this) {
            if (!loaded.get()) {
               loaded.set(true);
               try {
                  com.github.jcrfsuite.util.CrfSuiteLoader.load();
               } catch (Exception e) {
                  throw new RuntimeException(e);
               }
            }
         }
      }
   }

}//END OF CrfSuiteLoader
