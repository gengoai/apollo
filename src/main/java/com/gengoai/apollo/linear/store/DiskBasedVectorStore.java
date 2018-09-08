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

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.Validation.notNullOrBlank;

/**
 * The type Indexed file store.
 *
 * @author David B. Bracewell
 */
public final class DiskBasedVectorStore implements VectorStore, Serializable, Loggable {
   private static final long serialVersionUID = 1L;
   private final Map<String, Long> keyOffsets;
   private final File vectorFile;
   private transient final Cache<String, NDArray> vectorCache;
   private int dimension = 50;
   private Measure queryMeasure;
   private int cacheSize;


   private DiskBasedVectorStore(File vectorFile,
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
    * Builder builder.
    *
    * @return the builder
    */
   public static Builder builder() {
      return new Builder();
   }

   /**
    * Convenience method for loading an indexed vector store
    *
    * @param resource the resource containing the vectors
    * @return the vector store
    * @throws IOException Something went wrong reading the store
    */
   public static VectorStore read(Resource resource) throws IOException {
      return builder().location(resource.asFile().orElseThrow(IllegalStateException::new)).build();
   }

   @Override
   public boolean containsKey(String s) {
      return keyOffsets.containsKey(s);
   }

   @Override
   public int dimension() {
      return dimension;
   }

   @Override
   public NDArray get(String s) {
      return vectorCache.get(s);
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
   public Set<String> keySet() {
      return Collections.unmodifiableSet(keyOffsets.keySet());
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

   @Override
   public int size() {
      return keyOffsets.size();
   }

   @Override
   public VectorStoreBuilder toBuilder() {
      return new Builder().location(vectorFile)
                          .cacheSize(cacheSize)
                          .measure(queryMeasure);
   }

   private void writeToIndexFile(File indexFile) throws IOException {
      try (CSVWriter writer = CSV.csv().writer(Resources.fromFile(indexFile).setIsCompressed(true))) {
         writer.write(dimension);
         for (Map.Entry<String, Long> entry : keyOffsets.entrySet()) {
            writer.write(entry.getKey(), entry.getValue());
         }
      }
   }

   /**
    * The type Builder.
    */
   public static class Builder extends VectorStoreBuilder {
      private File location;
      private File tempLocation;
      private Map<String, Long> offsets = new HashMap<>();
      private RandomAccessFile writer;
      private long lastOffset = 0;
      private int cacheSize = 5_000;

      /**
       * Instantiates a new Builder.
       */
      public Builder() {
         this.location = Resources.temporaryFile().asFile().orElseThrow(IllegalStateException::new);
         this.tempLocation = Resources.temporaryFile().asFile().orElseThrow(IllegalStateException::new);
         try {
            this.writer = new RandomAccessFile(tempLocation, "rw");
         } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
         }
      }

      @Override
      public VectorStoreBuilder add(String key, NDArray vector) {
         notNullOrBlank(key, "The key must not be null or blank");
         try {
            if (dimension() == -1) {
               dimension = (int) vector.length();
               writer.write("? ".getBytes());
               writer.write(Integer.toString(dimension).getBytes());
               writer.write("\n".getBytes());
               lastOffset = writer.getFilePointer();
            }
            checkArgument(dimension == vector.length(),
                          () -> "Dimension mismatch. (" + dimension + ") != (" + vector.length() + ")");
            StringBuilder cLine = new StringBuilder(key);
            for (int i = 0; i < vector.length(); i++) {
               cLine.append(" ").append(vector.get(i));
            }
            cLine.append("\n");
            writer.write(cLine.toString().getBytes());
            offsets.put(key, lastOffset);
            lastOffset = writer.getFilePointer();
         } catch (IOException e) {
            throw new RuntimeException(e);
         }
         return this;
      }

      @Override
      public VectorStore build() throws IOException {
         writer.close();
         System.out.println(dimension);
         System.out.println(offsets);
         Resources.fromFile(tempLocation).copy(Resources.fromFile(location));
         File indexFile = new File(location.getAbsolutePath() + ".idx");
         try (CSVWriter writer = CSV.csv().writer(Resources.fromFile(indexFile).setIsCompressed(true))) {
            writer.write(dimension);
            for (Map.Entry<String, Long> entry : offsets.entrySet()) {
               writer.write(entry.getKey(), entry.getValue());
            }
         }
         return new DiskBasedVectorStore(location, cacheSize, measure());
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
   }

}//END OF DiskBasedVectorStore
