package com.gengoai.apollo.linear.store;

import com.gengoai.Parameters;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.cache.AutoCalculatingLRUCache;
import com.gengoai.cache.Cache;
import com.gengoai.io.IndexedFile;
import com.gengoai.io.IndexedFileReader;
import com.gengoai.io.IndexedFileWriter;
import com.gengoai.io.Resources;
import com.gengoai.io.resource.Resource;
import com.gengoai.logging.Loggable;
import com.gengoai.string.StringUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.Collections;
import java.util.Iterator;
import java.util.Set;

import static com.gengoai.Parameters.params;
import static com.gengoai.Validation.checkArgument;
import static com.gengoai.Validation.notNullOrBlank;
import static com.gengoai.apollo.linear.NDArray.vec2String;

/**
 * The type Indexed file store.
 *
 * @author David B. Bracewell
 */
public final class DiskBasedVectorStore implements VectorStore, Serializable, Loggable {
   private static final long serialVersionUID = 1L;
   private final IndexedFileReader reader;
   private transient final Cache<String, NDArray> vectorCache;
   private final int dimension;
   private final int cacheSize;


   public static void main(String[] args) throws Exception {
      VSBuilder builder = VectorStore.builder(params(VSParameter.IN_MEMORY, false,
                                                     VSParameter.CACHE_SIZE, 20_000,
                                                     VSParameter.LOCATION, "/home/ik/tmp.vec.txt"));
//      for (int i = 0; i < 10000; i++) {
//         builder.add(StringUtils.randomHexString(10), NDArrayFactory.DENSE.create(NDArrayInitializer.rand, 50));
//      }
      builder.build().forEach(n -> System.out.println(n.getLabel() + " : " + n));
   }

   public DiskBasedVectorStore(File vectorFile, int cacheSize) {
      try {
         this.cacheSize = cacheSize;
         this.dimension = determineDimension(vectorFile);
         this.reader = IndexedFile.reader(vectorFile, line -> line.split("[ \t]+")[0]);
         this.vectorCache = new AutoCalculatingLRUCache<>(cacheSize, this::loadNDArray);
      } catch (Exception e) {
         throw new RuntimeException(e);
      }
   }


   private static int determineDimension(File vectorStore) throws IOException {
      Resource r = Resources.fromFile(vectorStore);
      try (BufferedReader reader = new BufferedReader(r.reader())) {
         String line = reader.readLine();
         if (StringUtils.isNullOrBlank(line)) {
            throw new IOException("Unexpected empty line at beginning of file: " + vectorStore);
         }
         String[] cells = line.trim().split("[ \t]+");
         if (cells.length > 4) {
            return cells.length - 1;
         }
         return Integer.parseInt(cells[1]);
      }
   }

   @Override
   public void write(Resource location) throws IOException {
      Resources.fromFile(reader.getBackingFile()).copy(location);
      Resources.fromFile(reader.getIndexFile()).copy(Resources.from(location.descriptor() + IndexedFile.INDEX_EXT));
   }


   @Override
   public boolean containsKey(String s) {
      return reader.containsKey(s);
   }

   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public NDArray get(String s) {
      return vectorCache.get(s);
   }

   @Override
   public Iterator<NDArray> iterator() {
      return new Iterator<NDArray>() {
         private final Iterator<String> itr = reader.keySet().iterator();

         @Override
         public boolean hasNext() {
            return itr.hasNext();
         }

         @Override
         public NDArray next() {
            return vectorCache.get(itr.next());
         }
      };
   }

   @Override
   public Set<String> keySet() {
      return Collections.unmodifiableSet(reader.keySet());
   }

   private NDArray loadNDArray(String key) {
      if (reader.containsKey(key)) {
         try {
            NDArray vector = NDArrayFactory.DENSE.zeros(1, dimension);
            String[] parts = reader.get(key).split("[ \t]+");
            for (int i = 1; i < parts.length; i++) {
               vector.set(i - 1, Double.parseDouble(parts[i]));
            }
            vector.setLabel(parts[0]);
            return vector;
         } catch (Exception e) {
            throw new RuntimeException(e);
         }
      }
      return NDArrayFactory.SPARSE.zeros(dimension);
   }

   @Override
   public int size() {
      return reader.numberOfKeys();
   }

   @Override
   public Parameters<VSParameter> getParameters() {
      return params(VSParameter.IN_MEMORY, false,
                    VSParameter.CACHE_SIZE, cacheSize,
                    VSParameter.LOCATION, reader.getBackingFile().getAbsolutePath());
   }

   /**
    * The type Builder.
    */
   public static class Builder implements VSBuilder {
      private volatile IndexedFileWriter writer = null;
      private int dimension = -1;
      private final Parameters<VSParameter> parameters;

      public Builder(Parameters<VSParameter> parameters) {
         this.parameters = parameters;
      }


      private void ensureWriter() {
         if (writer == null) {
            synchronized (this) {
               if (writer == null) {
                  try {
                     writer = new IndexedFileWriter(new File(parameters.getString(VSParameter.LOCATION)));
                  } catch (IOException e) {
                     throw new RuntimeException(e);
                  }
               }
            }
         }
      }


      @Override
      public VSBuilder add(String key, NDArray vector) {
         notNullOrBlank(key, "The key must not be null or blank");
         try {
            if (dimension == -1) {
               dimension = (int) vector.length();
            }
            checkArgument(dimension == vector.length(),
                          () -> "Dimension mismatch. (" + dimension + ") != (" + vector.length() + ")");
            ensureWriter();
            writer.write(key, key + " " + vec2String(vector));
         } catch (IOException e) {
            throw new RuntimeException(e);
         }
         return this;
      }

      @Override
      public VectorStore build() {
         if (writer != null) {
            try {
               writer.close();
            } catch (IOException e) {
               throw new RuntimeException(e);
            }
         }
         return new DiskBasedVectorStore(new File(parameters.getString(VSParameter.LOCATION)),
                                         parameters.getInt(VSParameter.CACHE_SIZE));
      }

   }

}//END OF DiskBasedVectorStore
