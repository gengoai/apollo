package com.gengoai.apollo.linear.store;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.stat.measure.Measure;
import com.gengoai.cache.AutoCalculatingLRUCache;
import com.gengoai.cache.Cache;
import com.gengoai.io.CSV;
import com.gengoai.io.CSVReader;
import com.gengoai.io.CSVWriter;
import com.gengoai.io.Resources;
import com.gengoai.io.resource.Resource;
import com.gengoai.logging.Loggable;
import com.gengoai.string.CharMatcher;

import java.io.*;
import java.util.*;

/**
 * The type Indexed file store.
 *
 * @author David B. Bracewell
 */
public final class IndexedFileStore implements VectorStore<String>, Serializable, Loggable {
   private static final long serialVersionUID = 1L;
   private final Map<String, Long> keyOffsets;
   private final File vectorFile;
   private int dimension = 50;
   private Measure queryMeasure;
   private transient final Cache<String, NDArray> vectorCache;
   private int cacheSize;

   private IndexedFileStore(File vectorFile,
                            int cacheSize,
                            Measure queryMeasure
                           ) throws IOException {
      this.vectorFile = vectorFile;
      this.queryMeasure = queryMeasure;
      this.keyOffsets = new HashMap<>();
      this.cacheSize = cacheSize;
      indexFile();
      vectorCache = new AutoCalculatingLRUCache<>(cacheSize, this::loadNDArray);
   }

   /**
    * Convenience method for loading an indexed vector store
    *
    * @param resource the resource containing the vectors
    * @return the vector store
    * @throws IOException Something went wrong reading the store
    */
   public static VectorStore<String> read(Resource resource) throws IOException {
      return builder().location(resource.asFile().orElseThrow(IllegalStateException::new)).build();
   }


   private NDArray loadNDArray(String key) {
      if (keyOffsets.containsKey(key)) {
         try (RandomAccessFile raf = new RandomAccessFile(vectorFile, "r")) {
            long offset = keyOffsets.get(key);
            NDArray vector = NDArrayFactory.DENSE.zeros(dimension);
            raf.seek(offset);
            String line = raf.readLine();
            String[] parts = line.split("[ \t]+");
            for (int i = 1; i < parts.length; i++) {
               vector.set(i - 1, Double.parseDouble(parts[i]));
            }
            vector.setLabel(parts[0]);
            return vector;
         } catch (Exception e) {
            e.printStackTrace();
         }
      }
      return NDArrayFactory.SPARSE.zeros(dimension);
   }


   private void readFromIndex(File indexFile) throws IOException {
      try (CSVReader reader = CSV.csv().reader(Resources.fromFile(indexFile))) {
         List<String> row;
         dimension = Integer.parseInt(reader.nextRow().get(0));
         while ((row = reader.nextRow()) != null) {
            if (row.size() >= 2) {
               keyOffsets.put(row.get(0), Long.parseLong(row.get(1)));
            }
         }
      }
   }

   private void writeToIndexFile(File indexFile) throws IOException {
      try (CSVWriter writer = CSV.csv().writer(Resources.fromFile(indexFile).setIsCompressed(true))) {
         writer.write(dimension);
         for (Map.Entry<String, Long> entry : keyOffsets.entrySet()) {
            writer.write(entry.getKey(), entry.getValue());
         }
      }
   }

   private void indexFile() throws IOException {
      File indexFile = new File(vectorFile.getAbsolutePath() + ".idx");
      if (indexFile.exists()) {
         try {
            readFromIndex(indexFile);
            return;
         } catch (Exception e) {
            logWarn("Error loading pre-computed index file {0}, going to reindex.", e);
         }
      }

      try (RandomAccessFile raf = new RandomAccessFile(vectorFile, "r")) {
         String line = raf.readLine();
         long start = raf.getFilePointer();
         String[] cells = line.split("[ \t]+");
         if (cells.length > 4) {
            dimension = cells.length - 1;
            keyOffsets.put(cells[0], start);
         } else {
            dimension = Integer.parseInt(cells[1]);
         }
         start = raf.getFilePointer();
         while ((line = raf.readLine()) != null) {
            int i = CharMatcher.WhiteSpace.findIn(line);
            if (i > 0) {
               keyOffsets.put(line.substring(0, i), start);
               start = raf.getFilePointer();
            }
         }
      }

      try {
         writeToIndexFile(indexFile);
      } catch (Exception e) {
         logInfo("Error creating a pre-computed index file {0}, ignoring.", e);
         Resources.fromFile(indexFile).delete();
      }
   }


   @Override
   public boolean containsKey(String s) {
      return keyOffsets.containsKey(s);
   }

   @Override
   public Iterator<NDArray> iterator() {
      return new Iterator<NDArray>() {
         private final Iterator<String> itr = keyOffsets.keySet().iterator();

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
   public VectorStoreBuilder<String> toBuilder() {
      return new Builder().location(vectorFile)
                          .cacheSize(cacheSize)
                          .measure(queryMeasure)
                          .dimension(dimension());
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
   public Set<String> keySet() {
      return Collections.unmodifiableSet(keyOffsets.keySet());
   }

   @Override
   public Measure getQueryMeasure() {
      return queryMeasure;
   }

   @Override
   public int size() {
      return keyOffsets.size();
   }

   /**
    * Builder builder.
    *
    * @return the builder
    */
   public static Builder builder() {
      return new Builder();
   }

   /**
    * The type Builder.
    */
   public static class Builder extends VectorStoreBuilder<String> {
      private File location;
      private int cacheSize = 5_000;

      /**
       * Instantiates a new Builder.
       */
      public Builder() {
         this.location = Resources.temporaryFile().asFile().orElseThrow(IllegalStateException::new);
      }

      /**
       * Location builder.
       *
       * @param location the location
       * @return the builder
       */
      public Builder location(File location) {
         this.location = location;
         return this;
      }

      /**
       * Cache size builder.
       *
       * @param size the size
       * @return the builder
       */
      public Builder cacheSize(int size) {
         this.cacheSize = size;
         return this;
      }

      @Override
      public VectorStore<String> build() throws IOException {
         if (!location.exists() || vectors.size() > 0) {
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
         }
         return new IndexedFileStore(location, cacheSize, measure());
      }
   }

}//END OF IndexedFileStore
