package com.davidbracewell.apollo.linear.store;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.apollo.stat.measure.Measure;
import com.davidbracewell.guava.common.base.Throwables;
import com.davidbracewell.guava.common.cache.CacheBuilder;
import com.davidbracewell.guava.common.cache.CacheLoader;
import com.davidbracewell.guava.common.cache.LoadingCache;
import com.davidbracewell.io.Resources;
import com.davidbracewell.string.StringUtils;
import lombok.NonNull;
import org.apache.mahout.math.map.OpenObjectLongHashMap;

import java.io.*;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ExecutionException;

/**
 * @author David B. Bracewell
 */
public class IndexedFileStore implements VectorStore<String>, Serializable {
   private static final long serialVersionUID = 1L;
   private final OpenObjectLongHashMap<String> keyOffsets;
   private final File vectorFile;
   private int dimension;
   private Measure queryMeasure;
   private transient final LoadingCache<String, NDArray> vectorCache;


   public IndexedFileStore(File vectorFile,
                           int cacheSize,
                           @NonNull Measure queryMeasure
                          ) throws IOException {
      this.vectorFile = vectorFile;
      this.queryMeasure = queryMeasure;
      this.keyOffsets = new OpenObjectLongHashMap<>();
      indexFile();
      vectorCache = CacheBuilder.newBuilder()
                                .maximumSize(cacheSize)
                                .build(new CacheLoader<String, NDArray>() {
                                   @Override
                                   public NDArray load(String s) throws Exception {
                                      if (keyOffsets.containsKey(s)) {
                                         try (RandomAccessFile raf = new RandomAccessFile(vectorFile, "r")) {
                                            long offset = keyOffsets.get(s);
                                            NDArray vector = NDArrayFactory.DENSE_FLOAT.zeros(dimension);
                                            raf.seek(offset);
                                            String line = raf.readLine();
                                            String[] parts = line.split("\\p{Z}+");
                                            for (int i = 1; i < parts.length; i++) {
                                               vector.set(i - 1, Double.parseDouble(parts[i]));
                                            }
                                            vector.setLabel(parts[0]);
                                            return vector;
                                         } catch (Exception e) {
                                            e.printStackTrace();
                                         }
                                      }
                                      return NDArrayFactory.DENSE_FLOAT.zeros(dimension);
                                   }
                                });
   }


   private void indexFile() throws IOException {
      try (RandomAccessFile raf = new RandomAccessFile(vectorFile, "r")) {
         String line = raf.readLine();
         if (StringUtils.isNullOrBlank(line)) {
            throw new IllegalStateException("First line should contain number of rows and dimension size");
         }
         dimension = Integer.parseInt(line.split("\\p{Z}+")[1]);

         long start = raf.getFilePointer();
         while ((line = raf.readLine()) != null) {
            String[] cells = line.split("\\p{Z}+");
            keyOffsets.put(cells[0], start);
            start = raf.getFilePointer();
         }
      }
   }


   @Override
   public boolean containsKey(String s) {
      return keyOffsets.containsKey(s);
   }

   @Override
   public Iterator<NDArray> iterator() {
      return new Iterator<NDArray>() {
         private final Iterator<String> itr = keyOffsets.keys().iterator();

         @Override
         public boolean hasNext() {
            return itr.hasNext();
         }

         @Override
         public NDArray next() {
            try {
               return vectorCache.get(itr.next());
            } catch (ExecutionException e) {
               throw Throwables.propagate(e);
            }
         }
      };
   }

   @Override
   public VectorStoreBuilder<String> toBuilder() {
      return new Builder().measure(queryMeasure).dimension(dimension());
   }

   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public NDArray get(String s) {
      try {
         return vectorCache.get(s);
      } catch (ExecutionException e) {
         throw Throwables.propagate(e);
      }
   }

   @Override
   public Collection<String> keys() {
      return Collections.unmodifiableCollection(keyOffsets.keys());
   }

   @Override
   public Measure getQueryMeasure() {
      return queryMeasure;
   }

   @Override
   public int size() {
      return keyOffsets.size();
   }

   public static Builder builder() {
      return new Builder();
   }

   public static class Builder extends VectorStoreBuilder<String> {
      private File location;
      private int cacheSize = 1000;

      public Builder location(@NonNull File location) {
         this.location = location;
         return this;
      }

      public Builder cacheSize(int size) {
         this.cacheSize = size;
         return this;
      }

      @Override
      public VectorStore<String> build() throws IOException {
         try (BufferedWriter writer = new BufferedWriter(Resources.fromFile(location).writer())) {
            writer.write(Integer.toString(vectors.size()));
            writer.write(" ");
            writer.write(Integer.toString(dimension()));
            writer.write("\n");
            for (Map.Entry<String, NDArray> entry : vectors.entrySet()) {
               StringBuilder cLine = new StringBuilder(entry.getKey());
               for (int i = 0; i < entry.getValue().length(); i++) {
                  cLine.append(" ").append(entry.getValue().get(i));
               }
               cLine.append("\n");
               writer.write(cLine.toString());
            }
         }
         return new IndexedFileStore(location, cacheSize, measure());
      }
   }

}//END OF IndexedFileStore