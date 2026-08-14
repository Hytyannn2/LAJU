export interface DemandAnchor {
  id: string;
  name: string;
  jajahan: string;
  lat: number;
  lng: number;
  weight: number; // Attractor weight (1-100)
  category: 'market' | 'school_uni' | 'hospital' | 'commercial' | 'govt' | 'industrial' | 'transit_hub';
}

export const DEMAND_ANCHORS: DemandAnchor[] = [
  // --- KOTA BHARU (Hub & Highest Density) ---
  { id: 'kb_01', name: 'Pasar Siti Khadijah', jajahan: 'Kota Bharu', lat: 6.1301, lng: 102.2392, weight: 95, category: 'market' },
  { id: 'kb_02', name: 'AEON Mall Kota Bharu', jajahan: 'Kota Bharu', lat: 6.1190, lng: 102.2475, weight: 98, category: 'commercial' },
  { id: 'kb_03', name: 'KB Mall', jajahan: 'Kota Bharu', lat: 6.1265, lng: 102.2460, weight: 88, category: 'commercial' },
  { id: 'kb_04', name: 'Hospital Raja Perempuan Zainab II (HRPZ II)', jajahan: 'Kota Bharu', lat: 6.1262, lng: 102.2415, weight: 90, category: 'hospital' },
  { id: 'kb_05', name: 'Kompleks SUK Kelantan (Kota Darulnaim)', jajahan: 'Kota Bharu', lat: 6.1172, lng: 102.2443, weight: 82, category: 'govt' },
  { id: 'kb_06', name: 'Universiti Sains Malaysia (HUSM Kubang Kerian)', jajahan: 'Kota Bharu', lat: 6.0858, lng: 102.2708, weight: 92, category: 'hospital' },
  { id: 'kb_07', name: 'Kawasan Perindustrian Pengkalan Chepa', jajahan: 'Kota Bharu', lat: 6.1620, lng: 102.2850, weight: 78, category: 'industrial' },
  { id: 'kb_08', name: 'Lapangan Terbang Sultan Ismail Petra (LTSIP)', jajahan: 'Kota Bharu', lat: 6.1678, lng: 102.2925, weight: 80, category: 'transit_hub' },
  { id: 'kb_09', name: 'Universiti Malaysia Kelantan (UMK Pengkalan Chepa)', jajahan: 'Kota Bharu', lat: 6.1645, lng: 102.2840, weight: 84, category: 'school_uni' },
  { id: 'kb_10', name: 'Stesen Bas Central Kota Bharu (Tesco)', jajahan: 'Kota Bharu', lat: 6.1180, lng: 102.2490, weight: 86, category: 'transit_hub' },
  { id: 'kb_11', name: 'Pasar Berek 12', jajahan: 'Kota Bharu', lat: 6.1230, lng: 102.2530, weight: 70, category: 'market' },
  { id: 'kb_12', name: 'SMK Sultan Ismail', jajahan: 'Kota Bharu', lat: 6.1150, lng: 102.2580, weight: 65, category: 'school_uni' },
  { id: 'kb_13', name: 'Wakaf Che Yeh Night Market', jajahan: 'Kota Bharu', lat: 6.0880, lng: 102.2370, weight: 94, category: 'market' },
  { id: 'kb_14', name: 'Pantai Cahaya Bulan (PCB)', jajahan: 'Kota Bharu', lat: 6.1910, lng: 102.2680, weight: 60, category: 'commercial' },

  // --- PASIR MAS ---
  { id: 'pm_01', name: 'Pasar Besar Pasir Mas', jajahan: 'Pasir Mas', lat: 6.0425, lng: 102.1415, weight: 76, category: 'market' },
  { id: 'pm_02', name: 'Stesen Keretapi Pasir Mas (KTM)', jajahan: 'Pasir Mas', lat: 6.0438, lng: 102.1408, weight: 72, category: 'transit_hub' },
  { id: 'pm_03', name: 'Hospital Pasir Mas', jajahan: 'Pasir Mas', lat: 6.0480, lng: 102.1320, weight: 75, category: 'hospital' },
  { id: 'pm_04', name: 'Kolej Kemahiran Tinggi MARA Pasir Mas', jajahan: 'Pasir Mas', lat: 6.0120, lng: 102.1640, weight: 68, category: 'school_uni' },
  { id: 'pm_05', name: 'Zon Bebas Cukai Rantau Panjang', jajahan: 'Pasir Mas', lat: 6.0220, lng: 101.9740, weight: 88, category: 'commercial' },
  { id: 'pm_06', name: 'Pasar Tasek Raja Pasir Mas', jajahan: 'Pasir Mas', lat: 6.0350, lng: 102.1480, weight: 74, category: 'market' },
  { id: 'pm_07', name: 'Klinik Kesihatan Tendong', jajahan: 'Pasir Mas', lat: 6.0810, lng: 102.1960, weight: 58, category: 'hospital' },
  { id: 'pm_08', name: 'SMK Lati Pasir Mas', jajahan: 'Pasir Mas', lat: 6.0320, lng: 102.1610, weight: 55, category: 'school_uni' },

  // --- TUMPAT ---
  { id: 'tp_01', name: 'Stesen Keretapi Wakaf Bharu (KTM Main Hub)', jajahan: 'Tumpat', lat: 6.1155, lng: 102.2045, weight: 85, category: 'transit_hub' },
  { id: 'tp_02', name: 'Pasar Awam Tumpat', jajahan: 'Tumpat', lat: 6.1980, lng: 102.1680, weight: 68, category: 'market' },
  { id: 'tp_03', name: 'Hospital Tumpat', jajahan: 'Tumpat', lat: 6.1940, lng: 102.1740, weight: 70, category: 'hospital' },
  { id: 'tp_04', name: 'Jeti Pengkalan Kubor (Zon Bebas Cukai)', jajahan: 'Tumpat', lat: 6.2305, lng: 102.1485, weight: 82, category: 'commercial' },
  { id: 'tp_05', name: 'SMK Wakaf Bharu', jajahan: 'Tumpat', lat: 6.1120, lng: 102.2010, weight: 58, category: 'school_uni' },
  { id: 'tp_06', name: 'Pasar Chabang Empat Tumpat', jajahan: 'Tumpat', lat: 6.1590, lng: 102.1760, weight: 64, category: 'market' },

  // --- BACHOK ---
  { id: 'bc_01', name: 'Universiti Malaysia Kelantan (UMK Bachok Main Campus)', jajahan: 'Bachok', lat: 6.0090, lng: 102.4045, weight: 90, category: 'school_uni' },
  { id: 'bc_02', name: 'Pasar Awam Jelawat', jajahan: 'Bachok', lat: 6.0080, lng: 102.3700, weight: 75, category: 'market' },
  { id: 'bc_03', name: 'Pasar Bandar Bachok', jajahan: 'Bachok', lat: 6.0640, lng: 102.3990, weight: 70, category: 'market' },
  { id: 'bc_04', name: 'Hospital Bachok (Gunong)', jajahan: 'Bachok', lat: 6.0350, lng: 102.3550, weight: 72, category: 'hospital' },
  { id: 'bc_05', name: 'Klinik Kesihatan Tangok', jajahan: 'Bachok', lat: 6.0020, lng: 102.4080, weight: 60, category: 'hospital' },
  { id: 'bc_06', name: 'IKBN Bachok', jajahan: 'Bachok', lat: 6.0420, lng: 102.3850, weight: 62, category: 'school_uni' },
  { id: 'bc_07', name: 'Pantai Irama Bachok', jajahan: 'Bachok', lat: 6.0630, lng: 102.4040, weight: 65, category: 'commercial' },

  // --- TANAH MERAH ---
  { id: 'tm_01', name: 'Pasar Besar Tanah Merah', jajahan: 'Tanah Merah', lat: 5.8085, lng: 102.1480, weight: 78, category: 'market' },
  { id: 'tm_02', name: 'Stesen Keretapi Tanah Merah (KTM)', jajahan: 'Tanah Merah', lat: 5.8095, lng: 102.1472, weight: 74, category: 'transit_hub' },
  { id: 'tm_03', name: 'Hospital Tanah Merah', jajahan: 'Tanah Merah', lat: 5.8140, lng: 102.1420, weight: 77, category: 'hospital' },
  { id: 'tm_04', name: 'IKTM / Kolej Vokasional Tanah Merah', jajahan: 'Tanah Merah', lat: 5.8190, lng: 102.1550, weight: 62, category: 'school_uni' },
  { id: 'tm_05', name: 'Pusat Perniagaan Bandar Tanah Merah', jajahan: 'Tanah Merah', lat: 5.8060, lng: 102.1495, weight: 72, category: 'commercial' },

  // --- PASIR PUTEH ---
  { id: 'pp_01', name: 'Pasar Awam Pasir Puteh', jajahan: 'Pasir Puteh', lat: 5.8340, lng: 102.4040, weight: 75, category: 'market' },
  { id: 'pp_02', name: 'Hospital Tengku Anis Pasir Puteh', jajahan: 'Pasir Puteh', lat: 5.8390, lng: 102.3980, weight: 74, category: 'hospital' },
  { id: 'pp_03', name: 'Kawasan Perindustrian Tok Bali', jajahan: 'Pasir Puteh', lat: 5.8920, lng: 102.4850, weight: 80, category: 'industrial' },
  { id: 'pp_04', name: 'Pelabuhan Perikanan LKIM Tok Bali', jajahan: 'Pasir Puteh', lat: 5.8980, lng: 102.4920, weight: 78, category: 'industrial' },
  { id: 'pp_05', name: 'Matrikulasi Kelantan (KPM Kok Lanas / Pasir Puteh)', jajahan: 'Pasir Puteh', lat: 5.8450, lng: 102.3820, weight: 70, category: 'school_uni' },

  // --- MACHANG ---
  { id: 'mc_01', name: 'Universiti Teknologi MARA (UiTM Kampus Machang)', jajahan: 'Machang', lat: 5.7590, lng: 102.2180, weight: 88, category: 'school_uni' },
  { id: 'mc_02', name: 'Pasar Awam Machang', jajahan: 'Machang', lat: 5.7650, lng: 102.2150, weight: 72, category: 'market' },
  { id: 'mc_03', name: 'Hospital Machang', jajahan: 'Machang', lat: 5.7680, lng: 102.2110, weight: 73, category: 'hospital' },
  { id: 'mc_04', name: 'Pusat Pengajian Pondok Sentosa', jajahan: 'Machang', lat: 5.7480, lng: 102.2280, weight: 55, category: 'school_uni' },

  // --- KUALA KRAI ---
  { id: 'kk_01', name: 'Hospital Sultan Ismail Petra (Hospital Baru Kuala Krai)', jajahan: 'Kuala Krai', lat: 5.5180, lng: 102.2080, weight: 82, category: 'hospital' },
  { id: 'kk_02', name: 'Pasar Bandar Kuala Krai', jajahan: 'Kuala Krai', lat: 5.5320, lng: 102.2030, weight: 72, category: 'market' },
  { id: 'kk_03', name: 'Stesen Keretapi Intercity Kuala Krai (KTM)', jajahan: 'Kuala Krai', lat: 5.5340, lng: 102.2010, weight: 74, category: 'transit_hub' },
  { id: 'kk_04', name: 'Kolej Yaacobiah Kuala Krai', jajahan: 'Kuala Krai', lat: 5.5250, lng: 102.2060, weight: 60, category: 'school_uni' },
  { id: 'kk_05', name: 'Klinik Kesihatan Dabong', jajahan: 'Kuala Krai', lat: 5.3810, lng: 101.9960, weight: 52, category: 'hospital' },
  { id: 'kk_06', name: 'Stesen Keretapi Dabong (KTM Train Transit)', jajahan: 'Kuala Krai', lat: 5.3790, lng: 101.9940, weight: 68, category: 'transit_hub' },

  // --- GUA MUSANG ---
  { id: 'gm_01', name: 'Stesen Keretapi Gua Musang (KTM Hub)', jajahan: 'Gua Musang', lat: 4.8820, lng: 101.9680, weight: 75, category: 'transit_hub' },
  { id: 'gm_02', name: 'Hospital Gua Musang', jajahan: 'Gua Musang', lat: 4.8770, lng: 101.9610, weight: 76, category: 'hospital' },
  { id: 'gm_03', name: 'Pasar Awam Bandar Baru Gua Musang', jajahan: 'Gua Musang', lat: 4.8850, lng: 101.9650, weight: 70, category: 'market' },
  { id: 'gm_04', name: 'Kompleks Pejabat Pentadbiran Kesedar Gua Musang', jajahan: 'Gua Musang', lat: 4.8790, lng: 101.9720, weight: 66, category: 'govt' },
  { id: 'gm_05', name: 'Pasar Chiku 5 FELDA', jajahan: 'Gua Musang', lat: 5.1050, lng: 102.1150, weight: 54, category: 'market' },

  // --- JELI ---
  { id: 'jl_01', name: 'Universiti Malaysia Kelantan (UMK Kampus Jeli)', jajahan: 'Jeli', lat: 5.6960, lng: 101.8710, weight: 86, category: 'school_uni' },
  { id: 'jl_02', name: 'Pasar Awam Pekan Jeli', jajahan: 'Jeli', lat: 5.6940, lng: 101.8430, weight: 68, category: 'market' },
  { id: 'jl_03', name: 'Hospital Jeli', jajahan: 'Jeli', lat: 5.6910, lng: 101.8480, weight: 70, category: 'hospital' },
  { id: 'jl_04', name: 'Politeknik Jeli Kelantan', jajahan: 'Jeli', lat: 5.7120, lng: 101.8650, weight: 72, category: 'school_uni' },
  { id: 'jl_05', name: 'Stesen Bas & Teksi Pekan Jeli', jajahan: 'Jeli', lat: 5.6950, lng: 101.8410, weight: 62, category: 'transit_hub' },

  // --- LOJING ---
  { id: 'lj_01', name: 'Pusat Pertanian & Pasar Sayur Pos Terisu Lojing', jajahan: 'Lojing', lat: 4.6720, lng: 101.4250, weight: 58, category: 'market' },
  { id: 'lj_02', name: 'Klinik Kesihatan Lojing Highlands', jajahan: 'Lojing', lat: 4.6550, lng: 101.4020, weight: 50, category: 'hospital' },
  { id: 'lj_03', name: 'SK Pos Brooke Lojing', jajahan: 'Lojing', lat: 4.6850, lng: 101.4480, weight: 45, category: 'school_uni' },
  { id: 'lj_04', name: 'Kompleks Pejabat JKKK Lojing', jajahan: 'Lojing', lat: 4.6610, lng: 101.4110, weight: 42, category: 'govt' },
];
