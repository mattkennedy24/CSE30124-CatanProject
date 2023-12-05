import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { CsvService } from '../service/csvData.service'; // Import your CsvService

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent implements OnInit {
  catanBoard: any;
  catanStatsData!: any[]; // Variable to hold catan_stats.csv data
  catanScoresData!: any[]; // Variable to hold catanScores.csv data

  // Data structure to hold settlement placement locations
  settlementSpots = {
    threeTileSpots: [
      [0, 1, 4], [2, 5, 6], [1, 2, 5], [3, 4, 8], [0, 3, 4], [4, 5, 9],
      [1, 4, 5], [5, 6, 10], [3, 7, 8], [4, 8, 9], [5, 9, 10], [6, 10, 11],
      [7, 8, 12], [8, 9, 13], [9, 10, 14], [10, 11, 15], [8, 12, 13],
      [9, 13, 14], [10, 14, 15], [12, 13, 16], [13, 14, 17], [14, 15, 8],
      [13, 16, 17], [14, 17, 18]
    ],
    twoTileSpots: [
      [0, 1], [1, 2], [0, 3], [2, 6], [6, 11], [11, 15], [15, 18],
      [18, 17], [16, 17], [12, 16], [7, 12], [3, 7]
    ],
    oneTileSpots: [0, 1, 2, 6, 11, 15, 18, 17, 16, 12, 7, 3]
  };

  constructor(
    private route: ActivatedRoute,
    private csvService: CsvService
  ) {}

  ngOnInit(): void {
    this.route.paramMap.subscribe(params => {
      this.catanBoard = params.get('catanBoard');
      console.log('User Board data:', this.catanBoard);

    });

    // Fetch and parse catanstats.csv data
    this.csvService.getCsvData('catanstats.csv').subscribe(
      (data) => {
        this.catanStatsData = data;
        console.log('Parsed catan_stats.csv data:', this.catanStatsData);
      },
      (error) => {
        console.error('Error fetching catan_stats.csv data:', error);
      }
    );

    // Fetch and parse catan_scores.csv data
    this.csvService.getCsvData('catan_scores.csv').subscribe(
      (data) => {
        
        this.catanScoresData = data;

        console.log('Parsed catanScores.csv data:', this.catanScoresData);
      },
      (error) => {
        console.error('Error fetching catanScores.csv data:', error);
      }
    );
  }

  // Now that we have imported the board and csv data, rank the best intitial settlement placement locations

  



}
