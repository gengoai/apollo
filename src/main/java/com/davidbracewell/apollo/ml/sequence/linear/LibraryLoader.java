package com.davidbracewell.apollo.ml.sequence.linear;

import com.github.jcrfsuite.util.CrfSuiteLoader;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author David B. Bracewell
 */
public enum LibraryLoader {
   INSTANCE;
   private volatile AtomicBoolean done = new AtomicBoolean(false);

   public void load() {
      if (!done.get()) {
         synchronized (this) {
            if (!done.get()) {
               done.set(true);
               try {
                  CrfSuiteLoader.load();
               } catch (Exception e) {
                  e.printStackTrace();
               }
            }
         }
      }
   }
}//END OF LibraryLoader
