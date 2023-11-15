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


}
